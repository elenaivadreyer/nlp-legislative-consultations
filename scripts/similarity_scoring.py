#!/usr/bin/env python3
"""
Geothermie lobbying impact analysis pipeline (robust version).

- Handles multiple Org_* columns per row
- Supports "general comments" rows (Paragraph == -1) by comparing against aggregated law text
- Produces a long results table: one row per (paragraph/org/comment)
- Provides summary helpers: org ranking, influenced paragraphs, general vs specific comparison, top impacts, style analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# sentence-transformers can be heavy; import is ok, model load happens in __init__
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data container (optional)
# -----------------------------
@dataclass(frozen=True)
class LobbyingImpactMetrics:
    artikel: int
    paragraph: int
    absatz: int
    lobby_org: str
    text_similarity: float
    semantic_similarity: float
    change_adoption_rate: float
    keyword_overlap: float
    overall_impact_score: float
    comment_length: int
    has_change: bool
    is_general_comment: bool


# -----------------------------
# Analyzer
# -----------------------------
class GeothermieImpactAnalyzer:
    """
    Specialized analyzer for Geothermie law lobbying impact.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        *,
        load_model: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        if self.model is None:
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")

    # -----------------------------
    # Prep
    # -----------------------------
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Identify Org_* columns and basic required columns.
        """
        required = ["Artikel", "Paragraph", "Absatz", "Gesetzestext_Entwurf_1", "Gesetzestext_Entwurf_2"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        org_columns = [c for c in df.columns if c.startswith("Org_")]
        if not org_columns:
            raise ValueError("No Org_* columns found. Expected columns like Org_1, Org_2, ...")

        df_clean = df.copy()
        print(f"Found {len(org_columns)} lobby organizations")
        return df_clean, org_columns

    # -----------------------------
    # Similarities / metrics
    # -----------------------------
    @staticmethod
    def _safe_str(x: object) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):  # nan
            return ""
        if pd.isna(x):
            return ""
        return str(x)

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Simple cosine similarity using TF-IDF.
        """
        if not text1 or not text2:
            return 0.0

        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
            X = vec.fit_transform([text1, text2])
            return float(cosine_similarity(X[0:1], X[1:2])[0][0])
        except Exception:
            return 0.0

    @staticmethod
    def calculate_change_adoption(original: str, revised: str, comment: str) -> float:
        """
        % of comment vocabulary (len>3) that is newly introduced vs original and appears in revised.
        """
        if not comment:
            return 0.0

        o = original.lower().split()
        r = revised.lower().split()
        c = comment.lower().split()

        original_words = set(o)
        revised_words = set(r)
        comment_words = {w for w in set(c) if len(w) > 3}

        suggested = comment_words - original_words
        if not suggested:
            return 0.0

        adopted = suggested & revised_words
        return len(adopted) / len(suggested)

    def calculate_keyword_overlap(self, revised: str, comment: str) -> float:
        """
        Extract top TF-IDF keywords from comment and see how many appear in revised.
        """
        if not comment:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50, min_df=1)
            X = vectorizer.fit_transform([comment, revised])
            feature_names = vectorizer.get_feature_names_out()

            comment_tfidf = X[0].toarray()[0]
            top_idx = comment_tfidf.argsort()[-15:][::-1]
            keywords = [feature_names[i] for i in top_idx if comment_tfidf[i] > 0]

            if not keywords:
                return 0.0

            revised_lower = revised.lower()
            overlap = sum(1 for kw in keywords if kw.lower() in revised_lower)
            return overlap / len(keywords)
        except Exception:
            return 0.0

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        # cosine similarity for 1D vectors
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def calculate_semantic_similarity_batch(
        self,
        originals: List[str],
        reviseds: List[str],
        comments: List[str],
    ) -> List[float]:
        """
        Batch semantic similarity:
        - change_vector = revised - original
        - score = max(0, cos(comment, change_vector))*0.6 + max(0, cos(comment, revised)-cos(comment, original))*0.4
        """
        if self.model is None:
            self._load_model()
        assert self.model is not None

        # encode all at once for speed
        emb_o = self.model.encode(originals, convert_to_tensor=False, show_progress_bar=False)
        emb_r = self.model.encode(reviseds, convert_to_tensor=False, show_progress_bar=False)
        emb_c = self.model.encode(comments, convert_to_tensor=False, show_progress_bar=False)

        scores: List[float] = []
        for o, r, c in zip(emb_o, emb_r, emb_c):
            change_vec = r - o
            sim_dir = self._cos_sim(c, change_vec)
            sim_to_r = self._cos_sim(c, r)
            sim_to_o = self._cos_sim(c, o)

            semantic_score = max(0.0, sim_dir) * 0.6 + max(0.0, sim_to_r - sim_to_o) * 0.4
            scores.append(max(0.0, float(semantic_score)))
        return scores

    # -----------------------------
    # Core analysis
    # -----------------------------
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Returns long table: one row per (row/org/comment).
        """
        if weights is None:
            weights = {
                "text_similarity": 0.20,
                "semantic_similarity": 0.35,
                "change_adoption": 0.25,
                "keyword_overlap": 0.20,
            }

        df_clean, org_cols = self.prepare_data(df)

        # Precompute aggregated law texts for general comments
        law_original = " ".join(
            df_clean.loc[df_clean["Paragraph"] != -1, "Gesetzestext_Entwurf_1"].dropna().astype(str).tolist()
        )
        law_revised = " ".join(
            df_clean.loc[df_clean["Paragraph"] != -1, "Gesetzestext_Entwurf_2"].dropna().astype(str).tolist()
        )

        rows_for_semantic = []  # tuples of (original, revised, comment) aligned with results index

        results: List[dict] = []
        total_comments = 0

        for _, row in df_clean.iterrows():
            artikel = row["Artikel"]
            paragraph = row["Paragraph"]
            absatz = row["Absatz"]

            is_general = bool(paragraph == -1)

            original = law_original if is_general else self._safe_str(row["Gesetzestext_Entwurf_1"])
            revised = law_revised if is_general else self._safe_str(row["Gesetzestext_Entwurf_2"])

            if not original and not revised:
                continue

            has_change = (original != revised) if (original and revised) else False

            for org in org_cols:
                comment = self._safe_str(row.get(org, ""))

                if not comment.strip():
                    continue

                total_comments += 1

                # cheap metrics now
                text_sim = self.calculate_text_similarity(comment, revised)
                adoption = self.calculate_change_adoption(original, revised, comment)
                keywords = self.calculate_keyword_overlap(revised, comment)

                # semantic later in batch
                rows_for_semantic.append((original, revised, comment))

                results.append(
                    {
                        "Artikel": int(artikel),
                        "Paragraph": int(paragraph),
                        "Absatz": int(absatz),
                        "Is_General_Comment": is_general,
                        "Lobby_Org": org,
                        "Text_Similarity": float(text_sim),
                        "Change_Adoption_Rate": float(adoption),
                        "Keyword_Overlap": float(keywords),
                        "Comment_Length": int(len(comment)),
                        "Has_Change": bool(has_change),
                    }
                )

        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("No comments found (all Org_* cells empty/NaN).")
            return results_df

        # Batch semantic similarity (aligned with results_df rows)
        originals = [t[0] for t in rows_for_semantic]
        reviseds = [t[1] for t in rows_for_semantic]
        comments = [t[2] for t in rows_for_semantic]
        semantic_scores = self.calculate_semantic_similarity_batch(originals, reviseds, comments)
        results_df["Semantic_Similarity"] = np.round(semantic_scores, 6)

        # Overall score
        results_df["Overall_Impact_Score"] = (
            weights["text_similarity"] * results_df["Text_Similarity"]
            + weights["semantic_similarity"] * results_df["Semantic_Similarity"]
            + weights["change_adoption"] * results_df["Change_Adoption_Rate"]
            + weights["keyword_overlap"] * results_df["Keyword_Overlap"]
        )

        # Rounding for display
        for c in ["Text_Similarity", "Semantic_Similarity", "Change_Adoption_Rate", "Keyword_Overlap", "Overall_Impact_Score"]:
            results_df[c] = results_df[c].round(4)

        # Paragraph ID (vectorized)
        results_df["Paragraph_ID"] = np.where(
            results_df["Is_General_Comment"],
            "General_Comments",
            "Art"
            + results_df["Artikel"].astype(str)
            + "_§"
            + results_df["Paragraph"].astype(str)
            + "_Abs"
            + results_df["Absatz"].astype(str),
        )

        print(f"\nProcessed {total_comments} comments from {results_df['Lobby_Org'].nunique()} organizations")
        print(f"  - General comments: {int(results_df['Is_General_Comment'].sum())}")
        print(f"  - Specific paragraph comments: {int((~results_df['Is_General_Comment']).sum())}")

        return results_df

    # -----------------------------
    # Reporting helpers
    # -----------------------------
    def rank_organizations(self, results_df: pd.DataFrame, min_comments: int = 2) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        impact_by_org = results_df.groupby("Lobby_Org").agg(
            Avg_Impact=("Overall_Impact_Score", "mean"),
            Total_Impact=("Overall_Impact_Score", "sum"),
            Max_Impact=("Overall_Impact_Score", "max"),
            Avg_Text_Sim=("Text_Similarity", "mean"),
            Avg_Semantic_Sim=("Semantic_Similarity", "mean"),
            Avg_Adoption=("Change_Adoption_Rate", "mean"),
            Avg_Keyword_Overlap=("Keyword_Overlap", "mean"),
            Avg_Comment_Length=("Comment_Length", "mean"),
            Total_Comment_Length=("Comment_Length", "sum"),
            Num_Comments=("Paragraph_ID", "count"),
        ).round(4)

        impact_by_org = impact_by_org[impact_by_org["Num_Comments"] >= min_comments]
        return impact_by_org.sort_values("Avg_Impact", ascending=False)

    def identify_influenced_paragraphs(self, results_df: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        impact_by_para = results_df.groupby("Paragraph_ID").agg(
            Avg_Impact=("Overall_Impact_Score", "mean"),
            Max_Impact=("Overall_Impact_Score", "max"),
            Total_Impact=("Overall_Impact_Score", "sum"),
            Num_Comments=("Lobby_Org", "count"),
            Has_Change=("Has_Change", "first"),
        ).round(4)

        return impact_by_para.sort_values("Avg_Impact", ascending=False)

    def analyze_general_vs_specific(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Always returns BOTH rows:
        - Specific Comments (Is_General_Comment=False)
        - General Comments  (Is_General_Comment=True)
        """
        if results_df.empty:
            return pd.DataFrame()

        comparison = results_df.groupby("Is_General_Comment").agg(
            Avg_Impact=("Overall_Impact_Score", "mean"),
            Median_Impact=("Overall_Impact_Score", "median"),
            Max_Impact=("Overall_Impact_Score", "max"),
            Avg_Semantic_Sim=("Semantic_Similarity", "mean"),
            Avg_Adoption=("Change_Adoption_Rate", "mean"),
            Avg_Length=("Comment_Length", "mean"),
            Num_Comments=("Lobby_Org", "count"),
        ).round(4)

        # Force both groups
        comparison = comparison.reindex([False, True])
        comparison.index = ["Specific Comments", "General Comments"]
        return comparison

    def get_general_comment_impacts(self, results_df: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        general = results_df[results_df["Is_General_Comment"] == True].copy()
        if general.empty:
            return pd.DataFrame()

        general_summary = general.groupby("Lobby_Org").agg(
            Impact_Score=("Overall_Impact_Score", "mean"),
            Max_Impact=("Overall_Impact_Score", "max"),
            Semantic_Similarity=("Semantic_Similarity", "mean"),
            Comment_Length=("Comment_Length", "first"),
        ).round(4)

        return general_summary.sort_values("Impact_Score", ascending=False)

    def get_top_impacts(self, results_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        cols = [
            "Paragraph_ID",
            "Lobby_Org",
            "Overall_Impact_Score",
            "Semantic_Similarity",
            "Change_Adoption_Rate",
            "Keyword_Overlap",
            "Text_Similarity",
            "Comment_Length",
            "Has_Change",
        ]
        cols = [c for c in cols if c in results_df.columns]
        return results_df.nlargest(top_n, "Overall_Impact_Score")[cols]

    def analyze_comment_styles(self, results_df: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        style = results_df.groupby("Lobby_Org").agg(
            Avg_Length=("Comment_Length", "mean"),
            Std_Length=("Comment_Length", "std"),
            Min_Length=("Comment_Length", "min"),
            Max_Length=("Comment_Length", "max"),
            Avg_Impact=("Overall_Impact_Score", "mean"),
            Num_Comments=("Paragraph_ID", "count"),
        ).round(2)

        # Avoid divide-by-zero
        style["Impact_Per_100_Chars"] = (style["Avg_Impact"] / (style["Avg_Length"].replace(0, np.nan) / 100)).round(4)
        return style.sort_values("Avg_Impact", ascending=False)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    file_path = 'notebooks/geothermie_gesetz_kommentare.xlsx'
    sheet_name = 'Gesetz + Kommentare'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # clean texts by removing "-\n"
    df = df.replace('-\n', '', regex=True)

    # after clean simple "\n"
    df = df.replace('\n', ' ', regex=True)

    analyzer = GeothermieImpactAnalyzer(load_model=True)
    results = analyzer.analyze_dataframe(df)

    print("\n=== GENERAL VS SPECIFIC COMMENTS ===")
    print(analyzer.analyze_general_vs_specific(results))

    print("\n=== GENERAL COMMENT IMPACTS BY ORGANIZATION ===")
    general_impacts = analyzer.get_general_comment_impacts(results)
    print(general_impacts if not general_impacts.empty else "No general comments found")

    print("\n=== TOP 10 INDIVIDUAL IMPACTS ===")
    print(analyzer.get_top_impacts(results, top_n=10))

    print("\n=== ORGANIZATION RANKINGS ===")
    print(analyzer.rank_organizations(results, min_comments=1))

    print("\n=== MOST INFLUENCED PARAGRAPHS ===")
    print(analyzer.identify_influenced_paragraphs(results))

    print("\n=== COMMENT STYLE ANALYSIS ===")
    print(analyzer.analyze_comment_styles(results))

    # results.to_csv("lobbying_impact_results.csv", index=False)
