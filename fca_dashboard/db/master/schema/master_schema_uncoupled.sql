-------------------------------
-- DROP EXISTING OBJECTS
-------------------------------
-- (Drop in reverse dependency order)
DROP TABLE IF EXISTS Quality_Control_Records CASCADE;
DROP TABLE IF EXISTS Quality_Control_Types CASCADE;
DROP TABLE IF EXISTS Control_Board_Images CASCADE;
DROP TABLE IF EXISTS Control_Board_Items CASCADE;
DROP TABLE IF EXISTS Maintenance_Costs CASCADE;
DROP TABLE IF EXISTS Maintenance CASCADE;
DROP TABLE IF EXISTS Equipment_Documents CASCADE;
DROP TABLE IF EXISTS Project_Documents CASCADE;
DROP TABLE IF EXISTS Document_Types CASCADE;
DROP TABLE IF EXISTS Equipment_Projects CASCADE;
DROP TABLE IF EXISTS Project_Phases CASCADE;
DROP TABLE IF EXISTS Projects CASCADE;
DROP TABLE IF EXISTS equipment_mappings CASCADE;
DROP TABLE IF EXISTS cost_mappings CASCADE;
DROP TABLE IF EXISTS attribute_mappings CASCADE;
DROP TABLE IF EXISTS classification_mappings CASCADE;
DROP TABLE IF EXISTS direct_mappings CASCADE;
DROP TABLE IF EXISTS pattern_rules CASCADE;
DROP TABLE IF EXISTS mapping_rules CASCADE;
DROP TABLE IF EXISTS Equipment_TCO CASCADE;
DROP TABLE IF EXISTS ASHRAE_Service_Life CASCADE;
DROP TABLE IF EXISTS Equipment_Costs CASCADE;
DROP TABLE IF EXISTS Equipment_Attributes CASCADE;
DROP TABLE IF EXISTS Attribute_Templates CASCADE;
DROP TABLE IF EXISTS Equipment CASCADE;
DROP TABLE IF EXISTS Locations CASCADE;
DROP TABLE IF EXISTS Equipment_Categories CASCADE;
DROP TABLE IF EXISTS MCAA_Classifications CASCADE;
DROP TABLE IF EXISTS CatalogSystem CASCADE;
DROP TABLE IF EXISTS MasterFormat CASCADE;
DROP TABLE IF EXISTS UniFormat CASCADE;
DROP TABLE IF EXISTS OmniClass CASCADE;

-------------------------------
-- 1. CLASSIFICATION TABLES
-------------------------------
CREATE TABLE OmniClass (
    OmniClassID SERIAL PRIMARY KEY,
    OmniClassCode VARCHAR(50) NOT NULL,
    OmniClassTitle VARCHAR(100) NOT NULL,
    OmniClassDescription TEXT
);

CREATE TABLE UniFormat (
    UniFormatID SERIAL PRIMARY KEY,
    UniFormatCode VARCHAR(50) NOT NULL,
    UniFormatTitle VARCHAR(100) NOT NULL,
    UniFormatDescription TEXT
);

CREATE TABLE MasterFormat (
    MasterFormatID SERIAL PRIMARY KEY,
    MasterFormatCode VARCHAR(50) NOT NULL,
    MasterFormatTitle VARCHAR(100) NOT NULL,
    MasterFormatDescription TEXT
);

CREATE TABLE CatalogSystem (
    CatalogID SERIAL PRIMARY KEY,
    CatalogCode VARCHAR(50) NOT NULL,
    CatalogTitle VARCHAR(100) NOT NULL,
    CatalogDescription TEXT,
    ExternalReference VARCHAR(100)  -- Reference to external catalog system
);

CREATE TABLE MCAA_Classifications (
    MCAAID SERIAL PRIMARY KEY,
    SystemCategory VARCHAR(100) NOT NULL,  -- e.g., HVAC Equipment
    SystemName VARCHAR(100) NOT NULL,        -- e.g., Boilers
    SubSystemType VARCHAR(100),              -- e.g., Hot Water
    SubSystemClassification VARCHAR(100),    -- e.g., Cast Iron Sectional
    EquipmentSize VARCHAR(50),               -- Size specification
    Notes TEXT
);

-------------------------------
-- 2. EQUIPMENT_CATEGORIES (Decoupled)
-------------------------------
CREATE TABLE Equipment_Categories (
    CategoryID SERIAL PRIMARY KEY,
    CategoryName VARCHAR(100) NOT NULL,
    CategoryDescription TEXT
);

-- Junction Tables for Classifications
CREATE TABLE Equipment_Categories_OmniClass (
    CategoryID INT NOT NULL,
    OmniClassID INT NOT NULL,
    PRIMARY KEY (CategoryID, OmniClassID),
    CONSTRAINT fk_eco_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_eco_omniclass FOREIGN KEY (OmniClassID)
        REFERENCES OmniClass(OmniClassID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_UniFormat (
    CategoryID INT NOT NULL,
    UniFormatID INT NOT NULL,
    PRIMARY KEY (CategoryID, UniFormatID),
    CONSTRAINT fk_ecu_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecu_unifornat FOREIGN KEY (UniFormatID)
        REFERENCES UniFormat(UniFormatID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_MasterFormat (
    CategoryID INT NOT NULL,
    MasterFormatID INT NOT NULL,
    PRIMARY KEY (CategoryID, MasterFormatID),
    CONSTRAINT fk_ecm_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecm_masterformat FOREIGN KEY (MasterFormatID)
        REFERENCES MasterFormat(MasterFormatID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_Catalog (
    CategoryID INT NOT NULL,
    CatalogID INT NOT NULL,
    PRIMARY KEY (CategoryID, CatalogID),
    CONSTRAINT fk_ecc_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecc_catalog FOREIGN KEY (CatalogID)
        REFERENCES CatalogSystem(CatalogID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_MCAA (
    CategoryID INT NOT NULL,
    MCAAID INT NOT NULL,
    PRIMARY KEY (CategoryID, MCAAID),
    CONSTRAINT fk_ecmc_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecmc_mcaa FOREIGN KEY (MCAAID)
        REFERENCES MCAA_Classifications(MCAAID) ON UPDATE CASCADE ON DELETE RESTRICT
);

-- Create indexes for performance on junction tables
CREATE INDEX idx_eco_omniclass ON Equipment_Categories_OmniClass(OmniClassID);
CREATE INDEX idx_ecu_unifornat ON Equipment_Categories_UniFormat(UniFormatID);
CREATE INDEX idx_ecm_masterformat ON Equipment_Categories_MasterFormat(MasterFormatID);
CREATE INDEX idx_ecc_catalog ON Equipment_Categories_Catalog(CatalogID);
CREATE INDEX idx_ecmc_mcaa ON Equipment_Categories_MCAA(MCAAID);

-------------------------------
-- 3. LOCATIONS
-------------------------------
CREATE TABLE Locations (
    LocationID SERIAL PRIMARY KEY,
    BuildingName VARCHAR(100),
    Floor VARCHAR(50),
    Room VARCHAR(50),
    OtherLocationInfo TEXT,
    XCoordinate DECIMAL(10,6),   -- Spatial coordinate
    YCoordinate DECIMAL(10,6)
);

-------------------------------
-- 4. MAPPING MODULE BASE TABLE
-------------------------------
-- mapping_rules is referenced by Equipment and later mapping tables.
CREATE TABLE mapping_rules (
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,   -- e.g., pattern, direct, classification, attribute, cost
    source_type VARCHAR(50),
    target_type VARCHAR(50),
    priority INT
);

-------------------------------
-- 5. EQUIPMENT
-------------------------------
CREATE TABLE Equipment (
    EquipmentID SERIAL PRIMARY KEY,
    CategoryID INT NOT NULL,
    LocationID INT NOT NULL,
    EquipmentTag VARCHAR(50) NOT NULL,
    Manufacturer VARCHAR(100),
    Model VARCHAR(100),
    SerialNumber VARCHAR(100),
    Capacity FLOAT,
    InstallDate DATE,
    Status VARCHAR(50),
    rule_id INT,  -- Reference to mapping_rules
    CONSTRAINT fk_equip_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_equip_location FOREIGN KEY (LocationID)
        REFERENCES Locations(LocationID) ON UPDATE CASCADE ON DELETE RESTRICT
);
-- Now add FK for rule_id
ALTER TABLE Equipment
  ADD CONSTRAINT fk_equip_mapping_rule FOREIGN KEY (rule_id)
    REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL;

-------------------------------
-- 6. EQUIPMENT ATTRIBUTES (Unified)
-------------------------------
CREATE TABLE Attribute_Templates (
    TemplateID SERIAL PRIMARY KEY,
    CategoryID INT NOT NULL,
    AttributeName VARCHAR(100) NOT NULL,
    Description TEXT,
    DefaultUnit VARCHAR(50),
    IsRequired BOOLEAN DEFAULT FALSE,
    ValidationRule TEXT,        -- Regex or range rule
    DataType VARCHAR(50),       -- e.g., numeric, string, date
    CONSTRAINT fk_at_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Attributes (
    EquipAttrID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    TemplateID INT,   -- Optional link to Attribute_Templates
    AttributeName VARCHAR(100) NOT NULL,
    AttributeValue TEXT,
    UnitOfMeasure VARCHAR(50),
    rule_id INT,
    CONSTRAINT fk_ea_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ea_template FOREIGN KEY (TemplateID)
        REFERENCES Attribute_Templates(TemplateID) ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT fk_ea_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL
);
CREATE INDEX idx_ea_equipment ON Equipment_Attributes(EquipmentID);

-------------------------------
-- 7. EQUIPMENT COSTS (Unified)
-------------------------------
CREATE TABLE Equipment_Costs (
    CostID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    CostDate DATE NOT NULL,
    CostType VARCHAR(50) NOT NULL,
    Amount NUMERIC(12,2) NOT NULL,
    Comments TEXT,
    rule_id INT,
    CONSTRAINT fk_ecosts_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecosts_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL
);
CREATE INDEX idx_ecosts_equipment ON Equipment_Costs(EquipmentID);

-------------------------------
-- 8. TOTAL COST OF OWNERSHIP
-------------------------------
CREATE TABLE ASHRAE_Service_Life (
    ServiceLifeID SERIAL PRIMARY KEY,
    EquipmentType VARCHAR(100) NOT NULL,
    MedianLifeExpectancy INT NOT NULL,
    ServiceTeamPriority VARCHAR(50),
    Notes TEXT
);

CREATE TABLE Equipment_TCO (
    TCOID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    ServiceLifeID INT NOT NULL,
    -- Condition Assessment
    AssetCondition INT CHECK (AssetCondition BETWEEN 1 AND 5),
    FailureLikelihood INT CHECK (FailureLikelihood BETWEEN 1 AND 5),
    AssetCriticality INT CHECK (AssetCriticality BETWEEN 1 AND 5),
    ConditionScore NUMERIC(5,2),
    RiskCategory VARCHAR(20),
    -- Service Life
    EstimatedServiceLifeYears INT,
    EstimatedReplacementDate DATE,
    LifecycleStatus VARCHAR(50),
    -- Cost Data
    FirstCost NUMERIC(12,2),
    AnnualMaintenanceCost NUMERIC(12,2),
    ReplacementCost NUMERIC(12,2),
    AnnualEnergyCost NUMERIC(12,2),
    LastAssessmentDate DATE,
    AssessedBy VARCHAR(100),
    Notes TEXT,
    CONSTRAINT fk_tco_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_tco_service_life FOREIGN KEY (ServiceLifeID)
        REFERENCES ASHRAE_Service_Life(ServiceLifeID) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE INDEX idx_tco_equipment ON Equipment_TCO(EquipmentID);

-------------------------------
-- 9. MAPPING MODULE DETAILS
-------------------------------
CREATE TABLE pattern_rules (
    pattern_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    pattern_regex TEXT NOT NULL,
    replacement_template TEXT,
    CONSTRAINT fk_pattern_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE direct_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_value VARCHAR(100) NOT NULL,
    target_value VARCHAR(100) NOT NULL,
    CONSTRAINT fk_direct_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE classification_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_classification VARCHAR(100) NOT NULL,
    target_classification VARCHAR(50) NOT NULL,
    classification_type VARCHAR(50) NOT NULL,  -- e.g., OmniClass, UniFormat, MasterFormat
    CONSTRAINT fk_classification_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE attribute_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_attribute VARCHAR(100) NOT NULL,
    target_attribute VARCHAR(100) NOT NULL,
    transformation_rule TEXT,
    CONSTRAINT fk_attribute_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE cost_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    cost_type VARCHAR(50) NOT NULL,  -- e.g., Purchase, Installation, Maintenance
    source_currency VARCHAR(10),
    target_currency VARCHAR(10),
    CONSTRAINT fk_cost_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE equipment_mappings (
    mapping_entry_id SERIAL PRIMARY KEY,  -- surrogate key added
    equipment_id INT NOT NULL,
    rule_id INT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2) NOT NULL,  -- value between 0 and 1
    CONSTRAINT fk_equipmap_equipment FOREIGN KEY (equipment_id)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_equipmap_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_equipmap_equipment ON equipment_mappings(equipment_id);
CREATE INDEX idx_equipmap_rule ON equipment_mappings(rule_id);

-------------------------------
-- 10. PROJECTS, PHASES, and EQUIPMENT_PROJECTS
-------------------------------
CREATE TABLE Projects (
    ProjectID SERIAL PRIMARY KEY,
    ProjectName VARCHAR(100) NOT NULL,
    ProjectStartDate DATE,
    ProjectEndDate DATE,
    ProjectDescription TEXT
);

CREATE TABLE Project_Phases (
    PhaseID SERIAL PRIMARY KEY,
    ProjectID INT NOT NULL,
    OmniClassPhaseCode VARCHAR(50),
    PhaseTitle VARCHAR(100),
    StartDate DATE,
    EndDate DATE,
    Description TEXT,
    CONSTRAINT fk_phase_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_phase_project ON Project_Phases(ProjectID);

CREATE TABLE Equipment_Projects (
    EquipmentID INT NOT NULL,
    ProjectID INT NOT NULL,
    RoleOrStatus VARCHAR(50),
    StartDate DATE,
    EndDate DATE,
    PRIMARY KEY (EquipmentID, ProjectID),
    CONSTRAINT fk_eproj_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_eproj_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE
);

-------------------------------
-- 11. DOCUMENTATION MANAGEMENT
-------------------------------
CREATE TABLE Document_Types (
    DocTypeID SERIAL PRIMARY KEY,
    TypeName VARCHAR(50) NOT NULL,
    Description TEXT,
    AllowedFileTypes VARCHAR(100)
);

CREATE TABLE Project_Documents (
    DocumentID SERIAL PRIMARY KEY,
    ProjectID INT NOT NULL,
    PhaseID INT,
    DocTypeID INT NOT NULL,
    DocumentName VARCHAR(100) NOT NULL,
    FilePath TEXT NOT NULL,
    FileType VARCHAR(50),
    UploadDate DATE,
    Version VARCHAR(20),
    UploadedBy VARCHAR(100),
    Description TEXT,
    CONSTRAINT fk_pd_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_pd_phase FOREIGN KEY (PhaseID)
        REFERENCES Project_Phases(PhaseID) ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT fk_pd_doctype FOREIGN KEY (DocTypeID)
        REFERENCES Document_Types(DocTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Documents (
    DocID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    DocTypeID INT NOT NULL,
    DocumentName VARCHAR(100) NOT NULL,
    FilePath TEXT NOT NULL,
    FileType VARCHAR(50),
    UploadDate DATE,
    Version VARCHAR(20),
    UploadedBy VARCHAR(100),
    Description TEXT,
    CONSTRAINT fk_ed_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ed_doctype FOREIGN KEY (DocTypeID)
        REFERENCES Document_Types(DocTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);

-------------------------------
-- 12. MAINTENANCE and MAINTENANCE_COSTS
-------------------------------
CREATE TABLE Maintenance (
    MaintenanceID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    MaintenanceDate DATE NOT NULL,
    WorkPerformed TEXT,
    Technician VARCHAR(100),
    NextDueDate DATE,
    Comments TEXT,
    CONSTRAINT fk_maintenance_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE Maintenance_Costs (
    MaintCostID SERIAL PRIMARY KEY,
    MaintenanceID INT NOT NULL,
    CostType VARCHAR(50) NOT NULL,
    Amount NUMERIC(12,2) NOT NULL,
    Comments TEXT,
    CONSTRAINT fk_maintcost_maintenance FOREIGN KEY (MaintenanceID)
        REFERENCES Maintenance(MaintenanceID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_maintcost_maintenance ON Maintenance_Costs(MaintenanceID);

-------------------------------
-- 13. CONTROL BOARD MANAGEMENT
-------------------------------
CREATE TABLE Control_Board_Items (
    ControlItemID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    ItemName VARCHAR(100) NOT NULL,
    Description TEXT,
    Location VARCHAR(100),
    SetPoint VARCHAR(50),
    NormalRange VARCHAR(50),
    Units VARCHAR(50),
    CONSTRAINT fk_cbi_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_cbi_equipment ON Control_Board_Items(EquipmentID);

CREATE TABLE Control_Board_Images (
    ImageID SERIAL PRIMARY KEY,
    ControlItemID INT NOT NULL,
    ImagePath TEXT NOT NULL,
    ImageType VARCHAR(50),
    CaptureDate DATE,
    Description TEXT,
    CONSTRAINT fk_cbi_images FOREIGN KEY (ControlItemID)
        REFERENCES Control_Board_Items(ControlItemID) ON UPDATE CASCADE ON DELETE CASCADE
);

-------------------------------
-- 14. QUALITY CONTROL MANAGEMENT
-------------------------------
CREATE TABLE Quality_Control_Types (
    QCTypeID SERIAL PRIMARY KEY,
    TypeName VARCHAR(100) NOT NULL,  -- e.g., Service Verification, Installation Check, Data Accuracy
    Description TEXT,
    Department VARCHAR(100),
    RequiresApproval BOOLEAN DEFAULT FALSE
);

CREATE TABLE Quality_Control_Records (
    QCID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    QCTypeID INT NOT NULL,
    Verified BOOLEAN,
    VerificationDate DATE,
    VerifiedBy VARCHAR(100),
    Notes TEXT,
    Status VARCHAR(50),  -- e.g., Pending, Approved, Failed
    ApprovedBy VARCHAR(100),
    ApprovalDate DATE,
    CONSTRAINT fk_qcr_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_qcr_qctype FOREIGN KEY (QCTypeID)
        REFERENCES Quality_Control_Types(QCTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE INDEX idx_qcr_equipment ON Quality_Control_Records(EquipmentID);
CREATE INDEX idx_qcr_qctype ON Quality_Control_Records(QCTypeID);

-------------------------------
-- INDEXES for Other Tables (if needed)
-------------------------------
CREATE INDEX idx_maintenance_equipment ON Maintenance(EquipmentID);

-------------------------------
-- VIEWS (Common Business Patterns)
-------------------------------

-- View joining Equipment with its full classification details and location.
CREATE OR REPLACE VIEW v_equipment_full_details AS
SELECT 
    e.EquipmentID,
    e.EquipmentTag,
    e.Manufacturer,
    e.Model,
    e.SerialNumber,
    e.Capacity,
    e.InstallDate,
    e.Status,
    ec.CategoryName,
    ec.CategoryDescription,
    oc.OmniClassCode,
    oc.OmniClassTitle,
    uf.UniFormatCode,
    uf.UniFormatTitle,
    mf.MasterFormatCode,
    mf.MasterFormatTitle,
    cs.CatalogCode,
    cs.CatalogTitle,
    mc.SystemCategory,
    mc.SystemName,
    mc.SubSystemType,
    mc.SubSystemClassification,
    mc.EquipmentSize,
    mc.Notes,
    l.BuildingName,
    l.Floor,
    l.Room,
    l.OtherLocationInfo,
    l.XCoordinate,
    l.YCoordinate
FROM Equipment e
JOIN Equipment_Categories ec ON e.CategoryID = ec.CategoryID
LEFT JOIN Equipment_Categories_OmniClass eco ON ec.CategoryID = eco.CategoryID
LEFT JOIN OmniClass oc ON eco.OmniClassID = oc.OmniClassID
LEFT JOIN Equipment_Categories_UniFormat ecu ON ec.CategoryID = ecu.CategoryID
LEFT JOIN UniFormat uf ON ecu.UniFormatID = uf.UniFormatID
LEFT JOIN Equipment_Categories_MasterFormat ecm ON ec.CategoryID = ecm.CategoryID
LEFT JOIN MasterFormat mf ON ecm.MasterFormatID = mf.MasterFormatID
LEFT JOIN Equipment_Categories_Catalog ecc ON ec.CategoryID = ecc.CategoryID
LEFT JOIN CatalogSystem cs ON ecc.CatalogID = cs.CatalogID
LEFT JOIN Equipment_Categories_MCAA ecmc ON ec.CategoryID = ecmc.CategoryID
LEFT JOIN MCAA_Classifications mc ON ecmc.MCAAID = mc.MCAAID
JOIN Locations l ON e.LocationID = l.LocationID;

-- View for equipment mapping details.
CREATE OR REPLACE VIEW v_equipment_mapping AS
SELECT 
    em.mapping_entry_id,
    e.EquipmentID,
    e.EquipmentTag,
    mr.rule_id,
    mr.rule_name,
    em.applied_at,
    em.confidence_score
FROM equipment_mappings em
JOIN Equipment e ON em.equipment_id = e.EquipmentID
JOIN mapping_rules mr ON em.rule_id = mr.rule_id;

-- View for project documents with related project and phase information.
CREATE OR REPLACE VIEW v_project_documents AS
SELECT 
    pd.DocumentID,
    p.ProjectID,
    p.ProjectName,
    pp.PhaseID,
    pp.PhaseTitle,
    pd.DocumentName,
    pd.FilePath,
    pd.FileType,
    pd.UploadDate,
    pd.Version,
    pd.UploadedBy,
    pd.Description
FROM Project_Documents pd
JOIN Projects p ON pd.ProjectID = p.ProjectID
LEFT JOIN Project_Phases pp ON pd.PhaseID = pp.PhaseID;

-- View for equipment documents.
CREATE OR REPLACE VIEW v_equipment_documents AS
SELECT 
    ed.DocID,
    e.EquipmentID,
    e.EquipmentTag,
    ed.DocumentName,
    ed.FilePath,
    ed.FileType,
    ed.UploadDate,
    ed.Version,
    ed.UploadedBy,
    ed.Description
FROM Equipment_Documents ed
JOIN Equipment e ON ed.EquipmentID = e.EquipmentID;

-- View for maintenance details.
CREATE OR REPLACE VIEW v_equipment_maintenance AS
SELECT 
    m.MaintenanceID,
    e.EquipmentID,
    e.EquipmentTag,
    m.MaintenanceDate,
    m.WorkPerformed,
    m.Technician,
    m.NextDueDate,
    m.Comments
FROM Maintenance m
JOIN Equipment e ON m.EquipmentID = e.EquipmentID;
