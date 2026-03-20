# Mineru OOM kill (870c0ad9)

## Problem
Document `870c0ad9` (Cardio slidedeck PDF) fails because the mineru PDF extraction process gets OOM-killed by the container runtime. No detailed stack trace — just "Ingestion failed".

## Root cause
The mineru backend runs inside the container with a fixed memory limit. Large or image-heavy PDFs exceed available RAM during extraction.

## Fix options
1. **Increase container memory** — raise the memory limit in docker-compose.yml for the rag-anywhere service
2. **Fallback extractor** — if mineru fails, fall back to a lighter PDF parser (e.g. PyMuPDF/fitz) that uses less memory but produces lower-quality output
3. **Pre-check PDF size** — before sending to mineru, check page count / file size and warn or split large PDFs
