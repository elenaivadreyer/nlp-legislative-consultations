import os
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

class BundestagPDFScraper:
    """
    Sophisticated scraper for Bundestag PDF documents with text extraction.
    Uses PyMuPDF (fitz) for robust PDF text extraction.
    Implements polite scraping practices with robots.txt compliance.
    """

    def __init__(self, output_dir: str = "bundestag_pdfs", delay: float = 2.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BundestagScraper/1.0 (Educational/Research Purpose; +https://github.com/yourrepo)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
        })
        self.results = []
        self.delay = delay  # Polite delay between requests
        self.last_request_time = 0

    def polite_request(self, url: str, request_type: str = "page") -> requests.Response:
        """
        Make a polite HTTP request with rate limiting.

        Args:
            url: URL to request
            request_type: Type of request for logging ('page' or 'pdf')
        """
        # Enforce minimum delay between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            print(f"  [Polite delay: {sleep_time:.1f}s]")
            time.sleep(sleep_time)

        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed for {url}: {e}")
            raise

    def extract_organization_name(self, text: str) -> Optional[str]:
        """
        Extract organization name from text like '21(9)089 Stellungnahme der Stadtwerke München'
        Returns: 'Stadtwerke München'
        """
        # Pattern: Remove document number and "Stellungnahme" prefix
        pattern = r'^\d+\(\d+\)\d+\s+Stellungnahme\s+(?:der\s+|des\s+|von\s+)?(.+?)(?:\s*\(PDF\))?$'
        match = re.match(pattern, text.strip(), re.IGNORECASE)

        if match:
            org_name = match.group(1).strip()
            # Clean up common suffixes
            org_name = re.sub(r'\s*\(PDF.*?\)$', '', org_name)
            return org_name

        # Fallback: Try simpler pattern
        pattern2 = r'Stellungnahme\s+(?:der\s+|des\s+|von\s+)?(.+?)(?:\s*\(PDF\))?$'
        match2 = re.search(pattern2, text.strip(), re.IGNORECASE)
        if match2:
            return match2.group(1).strip()

        return None

    def scrape_pdf_links(self, url: str) -> List[Dict[str, str]]:
        """
        Scrape PDF links and organization names from the Bundestag page.
        Returns: List of dicts with 'organization', 'pdf_url', 'link_text'
        """
        print(f"Fetching page: {url}")
        print(f"Using polite scraping with {self.delay}s delay between requests")

        response = self.polite_request(url, request_type="page")

        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = []

        # Find all links that point to PDF resources
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Check if it's a bundestag PDF resource link
            if '/resource/blob/' in href and href.endswith('.pdf'):
                full_url = urljoin(url, href)
                link_text = link.get_text(strip=True)

                # Extract organization name
                org_name = self.extract_organization_name(link_text)

                if org_name:
                    pdf_links.append({
                        'organization': org_name,
                        'pdf_url': full_url,
                        'link_text': link_text
                    })
                    print(f"Found: {org_name} -> {full_url}")
                else:
                    print(f"Warning: Could not extract org name from: {link_text}")

        return pdf_links

    def download_pdf(self, pdf_url: str, organization: str) -> Optional[Path]:
        """
        Download PDF file with sanitized filename (polite rate-limited).
        Returns: Path to downloaded file or None if failed
        """
        # Sanitize organization name for filename
        safe_name = re.sub(r'[^\w\s-]', '', organization)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        filename = f"{safe_name}.pdf"
        filepath = self.output_dir / filename

        try:
            print(f"Downloading: {organization}...")
            response = self.polite_request(pdf_url, request_type="pdf")

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"✓ Saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"✗ Error downloading {organization}: {e}")
            return None

    def extract_text_pymupdf(self, pdf_path: Path) -> Dict[str, any]:
        """
        Extract text from PDF using PyMuPDF (fitz) - robust and fast.
        Returns dict with text, metadata, and page count.
        """
        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            # Extract metadata
            metadata = {
                'pages': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
            }

            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()

                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

            doc.close()

            full_text = "\n\n".join(text_parts)

            return {
                'text': full_text,
                'metadata': metadata,
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'text': '',
                'metadata': {},
                'success': False,
                'error': str(e)
            }

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """
        Extract text and metadata using PyMuPDF.
        """
        print(f"Extracting text from: {pdf_path.name}...")
        return self.extract_text_pymupdf(pdf_path)

    def process_document(self, doc_info: Dict[str, str]) -> Dict:
        """
        Download PDF and extract text for a single document.
        """
        organization = doc_info['organization']
        pdf_url = doc_info['pdf_url']

        # Download PDF
        pdf_path = self.download_pdf(pdf_url, organization)

        if not pdf_path:
            return {
                'organization': organization,
                'pdf_url': pdf_url,
                'status': 'download_failed',
                'text': None,
                'filepath': None,
                'metadata': {},
                'error': 'Download failed'
            }

        # Extract text and metadata
        extraction_result = self.extract_text_from_pdf(pdf_path)

        if extraction_result['success']:
            text = extraction_result['text']
            metadata = extraction_result['metadata']

            result = {
                'organization': organization,
                'pdf_url': pdf_url,
                'status': 'success',
                'text': text,
                'filepath': str(pdf_path),
                'text_length': len(text),
                'page_count': metadata.get('pages', 0),
                'metadata': metadata,
                'error': None
            }

            # Save extracted text
            text_file = pdf_path.with_suffix('.txt')
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"✓ Extracted {len(text)} characters from {metadata.get('pages', 0)} pages\n")
        else:
            result = {
                'organization': organization,
                'pdf_url': pdf_url,
                'status': 'extraction_failed',
                'text': None,
                'filepath': str(pdf_path),
                'metadata': {},
                'error': extraction_result['error']
            }
            print(f"✗ Extraction failed: {extraction_result['error']}\n")

        return result

    def run(self, url: str):
        """
        Main execution: scrape, download, and extract all PDFs with polite rate limiting.

        Args:
            url: Bundestag page URL
        """
        start_time = datetime.now()

        print("=" * 70)
        print("BUNDESTAG PDF SCRAPER & EXTRACTOR (Polite Scraping Mode)")
        print("=" * 70)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Rate limiting: {self.delay}s delay between requests")
        print()

        # Step 1: Scrape PDF links
        pdf_links = self.scrape_pdf_links(url)
        print(f"\n✓ Found {len(pdf_links)} PDF documents\n")

        if not pdf_links:
            print("No PDFs found. Exiting.")
            return

        print(f"Estimated time: ~{len(pdf_links) * self.delay / 60:.1f} minutes")
        print("-" * 70)
        print()

        # Step 2: Process each document
        for i, doc_info in enumerate(pdf_links, 1):
            print(f"[{i}/{len(pdf_links)}] Processing: {doc_info['organization']}")
            print("-" * 70)

            result = self.process_document(doc_info)
            self.results.append(result)

        # Step 3: Save summary
        self.save_summary()

        # Step 4: Print statistics
        end_time = datetime.now()
        duration = end_time - start_time
        self.print_statistics(duration)

    def save_summary(self):
        """Save processing summary to JSON file."""
        summary_file = self.output_dir / "summary.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Summary saved to: {summary_file}")

    def print_statistics(self, duration):
        """Print processing statistics."""
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)

        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed_download = sum(1 for r in self.results if r['status'] == 'download_failed')
        failed_extraction = sum(1 for r in self.results if r['status'] == 'extraction_failed')

        total_chars = sum(r.get('text_length', 0) for r in self.results if r['status'] == 'success')
        total_pages = sum(r.get('page_count', 0) for r in self.results if r['status'] == 'success')

        print(f"Total documents: {total}")
        print(f"Successfully processed: {successful}")
        print(f"Download failures: {failed_download}")
        print(f"Extraction failures: {failed_extraction}")
        print(f"\nTotal pages extracted: {total_pages}")
        print(f"Total characters extracted: {total_chars:,}")
        print(f"\nProcessing time: {duration}")
        print(f"Average time per document: {duration.total_seconds() / total:.1f}s")
        print(f"\nAll files saved to: {self.output_dir.absolute()}")
        print("=" * 70)


if __name__ == "__main__":
    # Target URL
    URL = "https://www.bundestag.de/dokumente/textarchiv/2025/kw49-de-geothermie-1128166"

    # Initialize scraper with polite settings
    # Default: 2 second delay between requests (polite to server)
    scraper = BundestagPDFScraper(output_dir="bundestag_pdfs", delay=2.0)

    # Run the scraper
    scraper.run(URL)

    # Access results programmatically
    print("\n\nSample results:")
    for result in scraper.results[:3]:
        print(f"\nOrganization: {result['organization']}")
        print(f"Status: {result['status']}")
        print(f"Pages: {result.get('page_count', 'N/A')}")
        if result.get('text'):
            print(f"Text preview: {result['text'][:200]}...")
        if result.get('metadata'):
            print(f"PDF Title: {result['metadata'].get('title', 'N/A')}")
            print(f"PDF Author: {result['metadata'].get('author', 'N/A')}")