-------------------------------
-- STAGING TO MASTER PROCEDURES
-------------------------------
-- These procedures handle the ETL process from staging to master tables

DROP SCHEMA IF EXISTS etl CASCADE;
CREATE SCHEMA etl;

CREATE OR REPLACE PROCEDURE etl.process_staging_data(
    batch_id VARCHAR DEFAULT NULL,
    limit_rows INTEGER DEFAULT 1000
)
LANGUAGE plpgsql
AS $$
DECLARE
    staging_rec RECORD;
    equipment_id INTEGER;
    category_id INTEGER;
    location_id INTEGER;
    cursor_count INTEGER := 0;
BEGIN
    -- Process a batch of records
    FOR staging_rec IN 
        SELECT * FROM staging.equipment_staging 
        WHERE (batch_id IS NULL OR import_batch_id = batch_id)
        AND processing_status = 'PENDING'
        LIMIT limit_rows
    LOOP
        BEGIN
            -- Mark as processing
            UPDATE staging.equipment_staging 
            SET processing_status = 'PROCESSING'
            WHERE staging_id = staging_rec.staging_id;
            
            -- Begin transaction for this record
            -- STEP 1: Process Equipment Category
            SELECT CategoryID INTO category_id
            FROM Equipment_Categories
            WHERE CategoryName = staging_rec.category_name;
            
            IF NOT FOUND THEN
                -- Create new category
                INSERT INTO Equipment_Categories(CategoryName, CategoryDescription)
                VALUES (staging_rec.category_name, 'Auto-created from staging')
                RETURNING CategoryID INTO category_id;
                -- Add classification mappings here
            END IF;
            
            -- STEP 2: Process Location
            SELECT LocationID INTO location_id
            FROM Locations
            WHERE BuildingName = staging_rec.building_name
            AND (Floor = staging_rec.floor OR (Floor IS NULL AND staging_rec.floor IS NULL))
            AND (Room = staging_rec.room OR (Room IS NULL AND staging_rec.room IS NULL));
            
            IF NOT FOUND THEN
                -- Create new location
                INSERT INTO Locations(BuildingName, Floor, Room, OtherLocationInfo, XCoordinate, YCoordinate)
                VALUES (
                    staging_rec.building_name,
                    staging_rec.floor,
                    staging_rec.room,
                    staging_rec.other_location_info,
                    staging_rec.x_coordinate,
                    staging_rec.y_coordinate
                )
                RETURNING LocationID INTO location_id;
            END IF;
            
            -- STEP 3: Process Equipment record
            SELECT EquipmentID INTO equipment_id
            FROM Equipment
            WHERE EquipmentTag = staging_rec.equipment_tag;
            
            IF NOT FOUND THEN
                -- Create new equipment
                INSERT INTO Equipment(
                    CategoryID,
                    LocationID,
                    EquipmentTag,
                    Manufacturer,
                    Model,
                    SerialNumber,
                    Capacity,
                    InstallDate,
                    Status,
                    rule_id
                )
                VALUES (
                    category_id,
                    location_id,
                    staging_rec.equipment_tag,
                    staging_rec.manufacturer,
                    staging_rec.model,
                    staging_rec.serial_number,
                    staging_rec.capacity,
                    staging_rec.install_date,
                    staging_rec.status,
                    staging_rec.mapping_rule_id
                )
                RETURNING EquipmentID INTO equipment_id;
            ELSE
                -- Update existing equipment
                UPDATE Equipment
                SET 
                    CategoryID = category_id,
                    LocationID = location_id,
                    Manufacturer = staging_rec.manufacturer,
                    Model = staging_rec.model,
                    SerialNumber = staging_rec.serial_number,
                    Capacity = staging_rec.capacity,
                    InstallDate = staging_rec.install_date,
                    Status = staging_rec.status,
                    rule_id = staging_rec.mapping_rule_id
                WHERE EquipmentID = equipment_id;
            END IF;
            
            -- STEP 4: Process attributes
            IF staging_rec.attributes IS NOT NULL THEN
                -- Process each attribute from JSON
                -- (simplified, would expand in real implementation)
            END IF;
            
            -- STEP 5: Process costs
            IF staging_rec.initial_cost IS NOT NULL THEN
                -- Insert or update costs
                -- (simplified, would expand in real implementation)
            END IF;
            
            -- Mark as processed
            UPDATE staging.equipment_staging 
            SET 
                processing_status = 'COMPLETED',
                is_processed = TRUE,
                processed_timestamp = CURRENT_TIMESTAMP
            WHERE staging_id = staging_rec.staging_id;
            
            cursor_count := cursor_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            -- Mark as error
            UPDATE staging.equipment_staging 
            SET 
                processing_status = 'ERROR',
                error_message = SQLERRM
            WHERE staging_id = staging_rec.staging_id;
        END;
    END LOOP;
    
    COMMIT;
    RAISE NOTICE 'Processed % records', cursor_count;
END;
$$;
