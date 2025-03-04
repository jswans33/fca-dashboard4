-------------------------------
-- MASTER TABLE SCHEMA FOR FCA DASHBOARD
-------------------------------
-- This schema uses a denormalized master table approach
-- that contains the most commonly used equipment data in one place,
-- with supporting tables for classifications and other related data.

DROP SCHEMA IF EXISTS master CASCADE;
CREATE SCHEMA master;

-------------------------------
-- CLASSIFICATION REFERENCE TABLES
-------------------------------
-- These tables provide reference for standardized classification systems

CREATE TABLE master.classification_systems (
    system_id SERIAL PRIMARY KEY,
    system_name VARCHAR(50) NOT NULL,  -- e.g., OmniClass, UniFormat, MasterFormat, MCAA
    system_version VARCHAR(50),
    description TEXT
);

CREATE TABLE master.classification_values (
    value_id SERIAL PRIMARY KEY,
    system_id INTEGER NOT NULL REFERENCES master.classification_systems(system_id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    parent_code VARCHAR(50),  -- For hierarchical classifications
    level INTEGER,            -- Hierarchy level
    description TEXT,
    UNIQUE(system_id, code)
);

-------------------------------
-- MASTER EQUIPMENT TABLE
-------------------------------
-- Core consolidated equipment table that contains all essential fields
-- This provides a simpler query interface for most common operations

CREATE TABLE master.equipment (
    equipment_id SERIAL PRIMARY KEY,
    
    -- Equipment identifiers
    equipment_tag VARCHAR(50) NOT NULL UNIQUE,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    capacity FLOAT,
    install_date DATE,
    status VARCHAR(50),
    
    -- Category & classification
    category_name VARCHAR(100) NOT NULL,
    omniclass_code VARCHAR(50),
    omniclass_title VARCHAR(100),
    uniformat_code VARCHAR(50),
    uniformat_title VARCHAR(100),
    masterformat_code VARCHAR(50),
    masterformat_title VARCHAR(100),
    catalog_code VARCHAR(50),
    
    -- MCAA classification data
    mcaa_system_category VARCHAR(100),
    mcaa_system_name VARCHAR(100),
    mcaa_subsystem_type VARCHAR(100),
    mcaa_subsystem_classification VARCHAR(100),
    mcaa_equipment_size VARCHAR(50),
    
    -- Location data
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    other_location_info TEXT,
    x_coordinate DECIMAL(10,6),
    y_coordinate DECIMAL(10,6),
    
    -- Asset condition & lifecycle data
    condition_score NUMERIC(5,2),  -- Overall condition (1-5 scale)
    risk_category VARCHAR(20),     -- e.g., High, Medium, Low
    expected_life_years INTEGER,
    estimated_replacement_date DATE,
    lifecycle_status VARCHAR(50),  -- e.g., New, Mid-Life, End-of-Life
    
    -- Cost data
    initial_cost NUMERIC(12,2),
    installation_cost NUMERIC(12,2),
    annual_maintenance_cost NUMERIC(12,2),
    replacement_cost NUMERIC(12,2),
    annual_energy_cost NUMERIC(12,2),
    last_cost_update DATE,
    
    -- Assessment data
    last_assessment_date DATE,
    assessed_by VARCHAR(100),
    
    -- Extended data (stored as JSON)
    attributes JSONB,           -- Flexible equipment attributes
    maintenance_summary JSONB,  -- Summary of recent maintenance
    documents_summary JSONB,    -- List of associated documents
    projects_summary JSONB,     -- List of associated projects
    
    -- Metadata and tracking
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    data_source VARCHAR(100),
    source_record_id VARCHAR(100),
    mapping_rule_id INTEGER,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for common query patterns
CREATE INDEX idx_master_equipment_tag ON master.equipment(equipment_tag);
CREATE INDEX idx_master_category ON master.equipment(category_name);
CREATE INDEX idx_master_building ON master.equipment(building_name);
CREATE INDEX idx_master_status ON master.equipment(status);
CREATE INDEX idx_master_lifecycle ON master.equipment(lifecycle_status);
CREATE INDEX idx_master_attributes ON master.equipment USING GIN (attributes);
CREATE INDEX idx_master_maintenance ON master.equipment USING GIN (maintenance_summary);
CREATE INDEX idx_master_documents ON master.equipment USING GIN (documents_summary);
CREATE INDEX idx_master_projects ON master.equipment USING GIN (projects_summary);

-------------------------------
-- SUPPORTING DETAIL TABLES
-------------------------------
-- These tables store detailed information that doesn't fit
-- well in the master table or needs separate management

-- Equipment attributes (for structured attribute storage)
CREATE TABLE master.equipment_attributes (
    attribute_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    attribute_name VARCHAR(100) NOT NULL,
    attribute_value TEXT,
    data_type VARCHAR(50),  -- e.g., text, number, date, boolean
    unit_of_measure VARCHAR(50),
    source_system VARCHAR(100),
    is_verified BOOLEAN DEFAULT FALSE,
    UNIQUE(equipment_id, attribute_name)
);

-- Maintenance records
CREATE TABLE master.maintenance_records (
    maintenance_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    maintenance_date DATE NOT NULL,
    maintenance_type VARCHAR(50),  -- e.g., Preventive, Corrective, Emergency
    description TEXT,
    performed_by VARCHAR(100),
    cost NUMERIC(12,2),
    parts_used TEXT,
    next_due_date DATE,
    status VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

-- Documents
CREATE TABLE master.documents (
    document_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    document_type VARCHAR(50) NOT NULL,  -- e.g., Manual, Warranty, Drawing, Photo
    document_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50),
    upload_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Projects
CREATE TABLE master.projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    description TEXT,
    start_date DATE,
    end_date DATE,
    status VARCHAR(50),
    budget NUMERIC(12,2),
    project_manager VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Equipment-project relationships
CREATE TABLE master.equipment_projects (
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES master.projects(project_id) ON DELETE CASCADE,
    relationship_type VARCHAR(50),  -- e.g., Installation, Replacement, Repair
    start_date DATE,
    end_date DATE,
    notes TEXT,
    PRIMARY KEY (equipment_id, project_id)
);

-------------------------------
-- AUDIT AND HISTORY TABLES
-------------------------------
-- These tables keep track of changes to equipment data

-- Equipment change history
CREATE TABLE master.equipment_history (
    history_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL,
    change_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    change_type VARCHAR(20) NOT NULL, -- INSERT