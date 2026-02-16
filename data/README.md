# Dataset

This directory contains documents for the RAG pipeline demo.

## Supported Formats

- **PDF files** (`.pdf`)
- **Text files** (`.txt`)

## Sample Dataset: Banking Regulations

The demo includes sample banking and financial regulation documents:

### Included Sample

**`basel_iii_overview.txt`** - Overview of Basel III banking regulations
- Source: Public domain summary of Basel Committee standards
- Topics: Capital requirements, leverage ratios, liquidity coverage

## Adding Your Own Documents

1. Place PDF or TXT files in this directory
2. Run the ingestion: `python -m src.main`
3. The pipeline will automatically:
   - Load documents
   - Split into chunks
   - Generate embeddings
   - Index for semantic search

## Public Financial Document Sources

If you want to add more documents, here are good sources:

### SEC Filings (US Public Companies)
- **URL:** https://www.sec.gov/edgar/searchedgar/companysearch.html
- **Files:** 10-K (annual reports), 10-Q (quarterly reports)
- **Format:** HTML or PDF
- **License:** Public domain

### Federal Reserve Publications
- **URL:** https://www.federalreserve.gov/publications.htm
- **Topics:** Monetary policy, banking supervision, financial stability
- **Format:** PDF
- **License:** Public domain

### Basel Committee Documents
- **URL:** https://www.bis.org/bcbs/
- **Topics:** Banking regulations, capital standards, risk management
- **Format:** PDF
- **License:** Public domain

### IMF Working Papers
- **URL:** https://www.imf.org/en/Publications/WP
- **Topics:** Economic policy, financial systems, country analysis
- **Format:** PDF
- **License:** Varies (check individual papers)

## Tips for Good Results

1. **Financial domain focus:** Use documents relevant to banking, finance, credit, risk
2. **Text quality:** Clean, well-formatted PDFs work best
3. **Document size:** 5-50 pages per document is ideal
4. **Quantity:** 5-20 documents gives good coverage without overwhelming
