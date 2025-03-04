-------------------------------
-- STAGING SCHEMA FOR FCA DASHBOARD
-------------------------------
-- This schema contains a single comprehensive staging table
-- to hold all incoming data before transformation and loading
-- into the master schema tables.

DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;

-- Central staging table for all incoming data
CREATE TABLE staging.equipment_staging (
    -- Staging metadata fields
    staging_id SERIAL PRIMARY KEY,
    source_system VARCHAR(100),
    import_batch_id VARCHAR(100),
    import_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, ERROR
    error_message TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    processed_timestamp TIMESTAMPTZ,
    
    -- Equipment fields
    equipment_tag VARCHAR(50),
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    capacity FLOAT,
    install_date DATE,
    status VARCHAR(50),
    
    -- Classification fields
    category_name VARCHAR(100),
    omniclass_code VARCHAR(50),
    omniclass_title VARCHAR(100),
    uniformat_code VARCHAR(50),
    uniformat_title VARCHAR(100),
    masterformat_code VARCHAR(50),
    masterformat_title VARCHAR(100),
    catalog_code VARCHAR(50),
    catalog_title VARCHAR(100),
    mcaa_system_category VARCHAR(100),
    mcaa_system_name VARCHAR(100),
    mcaa_subsystem_type VARCHAR(100),
    mcaa_subsystem_classification VARCHAR(100),
    mcaa_equipment_size VARCHAR(50),
    
    -- Location fields
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    other_location_info TEXT,
    x_coordinate DECIMAL(10,6),
    y_coordinate DECIMAL(10,6),
    
    -- Attribute fields (flexible JSON structure for varying attributes)
    attributes JSONB,
    
    -- Cost fields
    initial_cost NUMERIC(12,2),
    installation_cost NUMERIC(12,2),
    annual_maintenance_cost NUMERIC(12,2),
    replacement_cost NUMERIC(12,2),
    annual_energy_cost NUMERIC(12,2),
    cost_date DATE,
    cost_comments TEXT,
    
    -- TCO fields
    asset_condition INT,
    failure_likelihood INT,
    asset_criticality INT,
    condition_score NUMERIC(5,2),
    risk_category VARCHAR(20),
    estimated_service_life_years INT,
    estimated_replacement_date DATE,
    lifecycle_status VARCHAR(50),
    last_assessment_date DATE,
    assessed_by VARCHAR(100),
    
    -- Service life info
    equipment_type VARCHAR(100),
    median_life_expectancy INT,
    service_team_priority VARCHAR(50),
    
    -- Maintenance fields
    maintenance_data JSONB, -- Flexible structure for maintenance records
    
    -- Project fields
    project_data JSONB, -- Flexible structure for project relationships
    
    -- Document fields
    document_data JSONB, -- Flexible structure for documents
    
    -- Quality control fields
    qc_data JSONB, -- Flexible structure for quality control records
    
    -- Source data fields (for troubleshooting)
    raw_source_data JSONB,
    source_file_name VARCHAR(255),
    source_record_id VARCHAR(100),
    
    -- Mapping fields
    mapping_rule_id INT,
    mapping_confidence_score DECIMAL(3,2)
);

-- Indexes for performance
CREATE INDEX idx_staging_equipment_tag ON staging.equipment_staging(equipment_tag);
CREATE INDEX idx_staging_status ON staging.equipment_staging(processing_status);
CREATE INDEX idx_staging_batch ON staging.equipment_staging(import_batch_id);
CREATE INDEX idx_staging_processed ON staging.equipment_staging(is_processed);
CREATE INDEX idx_staging_attributes ON staging.equipment_staging USING GIN (attributes);
CREATE INDEX idx_staging_maintenance_data ON staging.equipment_staging USING GIN (maintenance_data);
CREATE INDEX idx_staging_project_data ON staging.equipment_staging USING GIN (project_data);
CREATE INDEX idx_staging_document_data ON staging.equipment_staging USING GIN (document_data);
CREATE INDEX idx_staging_qc_data ON staging.equipment_staging USING GIN (qc_data);

-- View for pending items
CREATE OR REPLACE VIEW staging.v_pending_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'PENDING';

-- View for errored items
CREATE OR REPLACE VIEW staging.v_error_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'ERROR';

-- Function to reset status of errored items
CREATE OR REPLACE FUNCTION staging.reset_error_items()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE staging.equipment_staging
    SET processing_status = 'PENDING',
        error_message = NULL
    WHERE processing_status = 'ERROR';
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clear staging table
CREATE OR REPLACE FUNCTION staging.clear_processed_items(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM staging.equipment_staging
    WHERE is_processed = TRUE 
    AND processed_timestamp < CURRENT_DATE - days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
