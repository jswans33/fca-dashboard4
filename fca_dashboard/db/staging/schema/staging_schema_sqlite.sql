-------------------------------
-- STAGING SCHEMA FOR FCA DASHBOARD (SQLite Version)
-------------------------------
-- This schema contains a single comprehensive staging table
-- to hold all incoming data before transformation and loading
-- into the master schema tables.

-- Central staging table for all incoming data
CREATE TABLE IF NOT EXISTS equipment_staging (
    -- Staging metadata fields
    staging_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_system TEXT,
    import_batch_id TEXT,
    import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status TEXT DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, ERROR
    error_message TEXT,
    is_processed INTEGER DEFAULT 0,  -- SQLite uses INTEGER for boolean (0=false, 1=true)
    processed_timestamp TIMESTAMP,
    
    -- Equipment fields
    equipment_tag TEXT,
    manufacturer TEXT,
    model TEXT,
    serial_number TEXT,
    capacity REAL,
    install_date DATE,
    status TEXT,
    
    -- Classification fields
    category_name TEXT,
    omniclass_code TEXT,
    omniclass_title TEXT,
    uniformat_code TEXT,
    uniformat_title TEXT,
    masterformat_code TEXT,
    masterformat_title TEXT,
    catalog_code TEXT,
    catalog_title TEXT,
    mcaa_system_category TEXT,
    mcaa_system_name TEXT,
    mcaa_subsystem_type TEXT,
    mcaa_subsystem_classification TEXT,
    mcaa_equipment_size TEXT,
    
    -- Location fields
    building_name TEXT,
    floor TEXT,
    room TEXT,
    other_location_info TEXT,
    x_coordinate REAL,
    y_coordinate REAL,
    
    -- Attribute fields (JSON structure for varying attributes)
    attributes TEXT,  -- SQLite doesn't have JSONB, using TEXT for JSON
    
    -- Cost fields
    initial_cost REAL,
    installation_cost REAL,
    annual_maintenance_cost REAL,
    replacement_cost REAL,
    annual_energy_cost REAL,
    cost_date DATE,
    cost_comments TEXT,
    
    -- TCO fields
    asset_condition INTEGER,
    failure_likelihood INTEGER,
    asset_criticality INTEGER,
    condition_score REAL,
    risk_category TEXT,
    estimated_service_life_years INTEGER,
    estimated_replacement_date DATE,
    lifecycle_status TEXT,
    last_assessment_date DATE,
    assessed_by TEXT,
    
    -- Service life info
    equipment_type TEXT,
    median_life_expectancy INTEGER,
    service_team_priority TEXT,
    
    -- Maintenance fields
    maintenance_data TEXT,  -- SQLite doesn't have JSONB, using TEXT for JSON
    
    -- Project fields
    project_data TEXT,  -- SQLite doesn't have JSONB, using TEXT for JSON
    
    -- Document fields
    document_data TEXT,  -- SQLite doesn't have JSONB, using TEXT for JSON
    
    -- Quality control fields
    qc_data TEXT,  -- SQLite doesn't have JSONB, using TEXT for JSON
    
    -- Source data fields (for troubleshooting)
    raw_source_data TEXT,  -- SQLite doesn't have JSONB, using TEXT for JSON
    source_file_name TEXT,
    source_record_id TEXT,
    
    -- Mapping fields
    mapping_rule_id INTEGER,
    mapping_confidence_score REAL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_staging_equipment_tag ON equipment_staging(equipment_tag);
CREATE INDEX IF NOT EXISTS idx_staging_status ON equipment_staging(processing_status);
CREATE INDEX IF NOT EXISTS idx_staging_batch ON equipment_staging(import_batch_id);
CREATE INDEX IF NOT EXISTS idx_staging_processed ON equipment_staging(is_processed);

-- SQLite doesn't support GIN indexes, so we can't create indexes on JSON fields
-- But we can create indexes on extracted JSON values if needed later

-- Create views for pending and error items
CREATE VIEW IF NOT EXISTS v_pending_items AS
SELECT * FROM equipment_staging 
WHERE processing_status = 'PENDING';

CREATE VIEW IF NOT EXISTS v_error_items AS
SELECT * FROM equipment_staging 
WHERE processing_status = 'ERROR';

-- SQLite doesn't support stored procedures/functions like PostgreSQL
-- We'll implement these functions in Python code instead