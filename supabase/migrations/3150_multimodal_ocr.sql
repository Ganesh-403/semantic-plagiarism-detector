-- Create processing state enums
CREATE TYPE ocr_processing_status AS ENUM ('queued', 'processing', 'aligned', 'failed');

-- Create Multimodal OCR Documents Table
CREATE TABLE ocr_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    storage_path TEXT NOT NULL,
    total_pages INT NOT NULL,
    status ocr_processing_status DEFAULT 'queued' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create Structural Layout & Token Extraction Ledger
CREATE TABLE ocr_extracted_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES ocr_documents(id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    raw_text TEXT NOT NULL,
    cleansed_text TEXT NOT NULL,
    bounding_box JSONB NOT NULL, -- Structure coordinates: {x0, y0, x1, y1}
    paraphrase_alignment_score NUMERIC(5, 2) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimize Indexing for Coordinate Scanning and Document Joins
CREATE INDEX idx_ocr_blocks_lookup ON ocr_extracted_blocks (document_id, page_number);
