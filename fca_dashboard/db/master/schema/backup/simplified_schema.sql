-------------------------------
-- SIMPLIFIED SCHEMA FOR FCA DASHBOARD
-------------------------------
-- Contains a simplified staging table and master table design

-- Create schemas
DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;

DROP SCHEMA IF EXISTS master CASCADE;
CREATE SCHEMA master;

-- Simplified staging table
CREATE TABLE staging.equipment_staging (
    -- Staging metadata fields
    staging_id SERIAL PRIMARY KEY,
    import_batch_id VARCHAR(100),
    import_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, PROCESSED, ERROR
    error_message TEXT,
    
    -- Core equipment fields
    equipment_tag VARCHAR(50),
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    status VARCHAR(50),
    
    -- Basic classification
    category_name VARCHAR(100),
    
    -- Simple location
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    
    -- Basic attributes
    attributes JSONB,
    
    -- Source data for troubleshooting
    source_system VARCHAR(100),
    raw_data JSONB
);

-- Indexes for staging table
CREATE INDEX idx_staging_status ON staging.equipment_staging(processing_status);
CREATE INDEX idx_staging_equipment_tag ON staging.equipment_staging(equipment_tag);

-- Master table for processed equipment data
CREATE TABLE master.equipment (
    equipment_id SERIAL PRIMARY KEY,
    equipment_tag VARCHAR(50) UNIQUE NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    status VARCHAR(50),
    category_name VARCHAR(100),
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    attributes JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    source_staging_id INT REFERENCES staging.equipment_staging(staging_id)
);

-- Indexes for master table
CREATE INDEX idx_master_equipment_tag ON master.equipment(equipment_tag);
CREATE INDEX idx_master_category ON master.equipment(category_name);
CREATE INDEX idx_master_building ON master.equipment(building_name);

-- Views
CREATE OR REPLACE VIEW staging.v_pending_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'PENDING';

CREATE OR REPLACE VIEW staging.v_error_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'ERROR';

-- Function to move data from staging to master
CREATE OR REPLACE FUNCTION staging.process_staging_data()
RETURNS INTEGER AS $$
DECLARE
    processed_count INTEGER := 0;
BEGIN
    -- Insert new equipment or update existing ones
    INSERT INTO master.equipment (
        equipment_tag, manufacturer, model, serial_number, 
        status, category_name, building_name, floor, room, 
        attributes, source_staging_id
    )
    SELECT 
        s.equipment_tag, s.manufacturer, s.model, s.serial_number,
        s.status, s.category_name, s.building_name, s.floor, s.room,
        s.attributes, s.staging_id
    FROM staging.equipment_staging s
    WHERE s.processing_status = 'PENDING'
    ON CONFLICT (equipment_tag) 
    DO UPDATE SET
        manufacturer = EXCLUDED.manufacturer,
        model = EXCLUDED.model,
        serial_number = EXCLUDED.serial_number,
        status = EXCLUDED.status,
        category_name = EXCLUDED.category_name,
        building_name = EXCLUDED.building_name,
        floor = EXCLUDED.floor,
        room = EXCLUDED.room,
        attributes = EXCLUDED.attributes,
        updated_at = CURRENT_TIMESTAMP,
        source_staging_id = EXCLUDED.source_staging_id;

    -- Mark records as processed
    UPDATE staging.equipment_staging
    SET processing_status = 'PROCESSED'
    WHERE processing_status = 'PENDING';
    
    GET DIAGNOSTICS processed_count = ROW_COUNT;
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clear processed records
CREATE OR REPLACE FUNCTION staging.clear_processed_items(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM staging.equipment_staging
    WHERE processing_status = 'PROCESSED' 
    AND import_timestamp < CURRENT_DATE - days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
