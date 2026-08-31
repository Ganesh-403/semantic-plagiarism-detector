-- Create Clone Matching Classification Enum
CREATE TYPE code_clone_type AS ENUM ('type_1_exact', 'type_2_renamed', 'type_3_restructured', 'type_4_semantic');

-- Create Code Submissions Table
CREATE TABLE code_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    assignment_id UUID NOT NULL,
    source_code TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create Semantic AST Fingerprints Ledger
CREATE TABLE code_ast_fingerprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES code_submissions(id) ON DELETE CASCADE,
    normalized_ast_hash CHAR(64) NOT NULL, -- SHA-256 structural signature
    structural_tokens JSONB NOT NULL,     -- Array of normalized node tags
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create Identified Code Clones Ledger
CREATE TABLE identified_code_clones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_submission_id UUID REFERENCES code_submissions(id) ON DELETE CASCADE,
    matched_submission_id UUID REFERENCES code_submissions(id) ON DELETE CASCADE,
    similarity_score NUMERIC(5, 2) NOT NULL,
    clone_classification code_clone_type NOT NULL,
    matched_blocks JSONB NOT NULL,          -- Line alignment map
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimize Indexes for Sub-second Fingerprint Intersection Queries
CREATE INDEX idx_ast_hash_lookup ON code_ast_fingerprints(normalized_ast_hash);
CREATE INDEX idx_clone_similarity_score ON identified_code_clones(similarity_score DESC);
