"""
SQLite procedures for transforming data from staging to master tables.

This module provides Python functions that replicate the functionality
of the PostgreSQL stored procedures for the ETL process.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database.base import DatabaseError
from fca_dashboard.utils.logging_config import get_logger

logger = get_logger("staging_to_master_sqlite")


def process_staging_data(
    connection_string: str,
    batch_id: Optional[str] = None,
    limit_rows: int = 1000
) -> Dict[str, Union[str, int]]:
    """
    Process data from the staging table to the master tables.
    
    Args:
        connection_string: The SQLite connection string.
        batch_id: Optional batch ID to filter records.
        limit_rows: Maximum number of rows to process in one batch.
        
    Returns:
        A dictionary containing the results of the processing.
        
    Raises:
        DatabaseError: If an error occurs during processing.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Initialize counters
        processed_count = 0
        error_count = 0
        
        # Get pending records
        with engine.connect() as conn:
            # Build the query
            query = """
                SELECT * FROM equipment_staging 
                WHERE processing_status = 'PENDING'
            """
            
            if batch_id:
                query += f" AND import_batch_id = '{batch_id}'"
                
            query += f" LIMIT {limit_rows}"
            
            # Execute the query
            result = conn.execute(text(query))
            staging_records = result.fetchall()
            
            # Process each record
            for record in staging_records:
                try:
                    # Mark as processing
                    update_status_query = f"""
                        UPDATE equipment_staging 
                        SET processing_status = 'PROCESSING'
                        WHERE staging_id = {record['staging_id']}
                    """
                    conn.execute(text(update_status_query))
                    conn.commit()
                    
                    # Process the record
                    _process_single_record(conn, record)
                    
                    # Mark as completed
                    complete_query = f"""
                        UPDATE equipment_staging 
                        SET 
                            processing_status = 'COMPLETED',
                            is_processed = 1,
                            processed_timestamp = '{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
                        WHERE staging_id = {record['staging_id']}
                    """
                    conn.execute(text(complete_query))
                    conn.commit()
                    
                    processed_count += 1
                    
                except Exception as e:
                    # Mark as error
                    error_message = str(e).replace("'", "''")
                    error_query = f"""
                        UPDATE equipment_staging 
                        SET 
                            processing_status = 'ERROR',
                            error_message = '{error_message}'
                        WHERE staging_id = {record['staging_id']}
                    """
                    conn.execute(text(error_query))
                    conn.commit()
                    
                    error_count += 1
                    logger.error(f"Error processing record {record['staging_id']}: {str(e)}")
        
        logger.info(f"Processed {processed_count} records, {error_count} errors")
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "error_count": error_count
        }
    except Exception as e:
        error_msg = f"Error processing staging data: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def _process_single_record(conn, record):
    """
    Process a single record from staging to master tables.
    
    Args:
        conn: The database connection.
        record: The staging record to process.
        
    Raises:
        Exception: If an error occurs during processing.
    """
    # STEP 1: Process Equipment Category
    category_id = _get_or_create_category(conn, record)
    
    # STEP 2: Process Location
    location_id = _get_or_create_location(conn, record)
    
    # STEP 3: Process Equipment record
    equipment_id = _get_or_create_equipment(conn, record, category_id, location_id)
    
    # STEP 4: Process attributes
    if record['attributes']:
        _process_attributes(conn, equipment_id, record)
    
    # STEP 5: Process costs
    if record['initial_cost'] is not None:
        _process_costs(conn, equipment_id, record)


def _get_or_create_category(conn, record):
    """Get or create an equipment category."""
    # Check if category exists
    query = f"""
        SELECT CategoryID FROM Equipment_Categories
        WHERE CategoryName = '{record['category_name']}'
    """
    result = conn.execute(text(query))
    category = result.fetchone()
    
    if category:
        return category['CategoryID']
    else:
        # Create new category
        insert_query = f"""
            INSERT INTO Equipment_Categories(CategoryName, CategoryDescription)
            VALUES ('{record['category_name']}', 'Auto-created from staging')
        """
        result = conn.execute(text(insert_query))
        conn.commit()
        
        # Get the new category ID
        query = "SELECT last_insert_rowid() as CategoryID"
        result = conn.execute(text(query))
        return result.fetchone()['CategoryID']


def _get_or_create_location(conn, record):
    """Get or create a location."""
    # Check if location exists
    floor_condition = "IS NULL" if record['floor'] is None else f"= '{record['floor']}'"
    room_condition = "IS NULL" if record['room'] is None else f"= '{record['room']}'"
    
    query = f"""
        SELECT LocationID FROM Locations
        WHERE BuildingName = '{record['building_name']}'
        AND (Floor {floor_condition})
        AND (Room {room_condition})
    """
    result = conn.execute(text(query))
    location = result.fetchone()
    
    if location:
        return location['LocationID']
    else:
        # Create new location
        floor_value = "NULL" if record['floor'] is None else f"'{record['floor']}'"
        room_value = "NULL" if record['room'] is None else f"'{record['room']}'"
        other_info_value = "NULL" if record['other_location_info'] is None else f"'{record['other_location_info']}'"
        x_coord_value = "NULL" if record['x_coordinate'] is None else f"{record['x_coordinate']}"
        y_coord_value = "NULL" if record['y_coordinate'] is None else f"{record['y_coordinate']}"
        
        insert_query = f"""
            INSERT INTO Locations(BuildingName, Floor, Room, OtherLocationInfo, XCoordinate, YCoordinate)
            VALUES (
                '{record['building_name']}',
                {floor_value},
                {room_value},
                {other_info_value},
                {x_coord_value},
                {y_coord_value}
            )
        """
        result = conn.execute(text(insert_query))
        conn.commit()
        
        # Get the new location ID
        query = "SELECT last_insert_rowid() as LocationID"
        result = conn.execute(text(query))
        return result.fetchone()['LocationID']


def _get_or_create_equipment(conn, record, category_id, location_id):
    """Get or create an equipment record."""
    # Check if equipment exists
    query = f"""
        SELECT EquipmentID FROM Equipment
        WHERE EquipmentTag = '{record['equipment_tag']}'
    """
    result = conn.execute(text(query))
    equipment = result.fetchone()
    
    # Prepare values with proper NULL handling
    manufacturer_value = "NULL" if record['manufacturer'] is None else f"'{record['manufacturer']}'"
    model_value = "NULL" if record['model'] is None else f"'{record['model']}'"
    serial_value = "NULL" if record['serial_number'] is None else f"'{record['serial_number']}'"
    capacity_value = "NULL" if record['capacity'] is None else f"{record['capacity']}"
    install_date_value = "NULL" if record['install_date'] is None else f"'{record['install_date']}'"
    status_value = "NULL" if record['status'] is None else f"'{record['status']}'"
    rule_id_value = "NULL" if record['mapping_rule_id'] is None else f"{record['mapping_rule_id']}"
    
    if equipment:
        # Update existing equipment
        update_query = f"""
            UPDATE Equipment
            SET 
                CategoryID = {category_id},
                LocationID = {location_id},
                Manufacturer = {manufacturer_value},
                Model = {model_value},
                SerialNumber = {serial_value},
                Capacity = {capacity_value},
                InstallDate = {install_date_value},
                Status = {status_value},
                rule_id = {rule_id_value}
            WHERE EquipmentID = {equipment['EquipmentID']}
        """
        conn.execute(text(update_query))
        conn.commit()
        return equipment['EquipmentID']
    else:
        # Create new equipment
        insert_query = f"""
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
                {category_id},
                {location_id},
                '{record['equipment_tag']}',
                {manufacturer_value},
                {model_value},
                {serial_value},
                {capacity_value},
                {install_date_value},
                {status_value},
                {rule_id_value}
            )
        """
        conn.execute(text(insert_query))
        conn.commit()
        
        # Get the new equipment ID
        query = "SELECT last_insert_rowid() as EquipmentID"
        result = conn.execute(text(query))
        return result.fetchone()['EquipmentID']


def _process_attributes(conn, equipment_id, record):
    """Process equipment attributes."""
    # Parse JSON attributes
    try:
        if isinstance(record['attributes'], str):
            attributes = json.loads(record['attributes'])
        else:
            attributes = record['attributes']
            
        # Process each attribute
        for key, value in attributes.items():
            # Check if attribute exists
            query = f"""
                SELECT AttributeID FROM Equipment_Attributes
                WHERE EquipmentID = {equipment_id} AND AttributeName = '{key}'
            """
            result = conn.execute(text(query))
            attribute = result.fetchone()
            
            value_str = str(value).replace("'", "''")
            
            if attribute:
                # Update existing attribute
                update_query = f"""
                    UPDATE Equipment_Attributes
                    SET AttributeValue = '{value_str}'
                    WHERE AttributeID = {attribute['AttributeID']}
                """
                conn.execute(text(update_query))
            else:
                # Create new attribute
                insert_query = f"""
                    INSERT INTO Equipment_Attributes(EquipmentID, AttributeName, AttributeValue)
                    VALUES ({equipment_id}, '{key}', '{value_str}')
                """
                conn.execute(text(insert_query))
                
        conn.commit()
    except Exception as e:
        logger.error(f"Error processing attributes: {str(e)}")
        raise Exception(f"Error processing attributes: {str(e)}")


def _process_costs(conn, equipment_id, record):
    """Process equipment costs."""
    try:
        # Check if cost record exists
        query = f"""
            SELECT CostID FROM Equipment_Costs
            WHERE EquipmentID = {equipment_id}
        """
        result = conn.execute(text(query))
        cost = result.fetchone()
        
        # Prepare values with proper NULL handling
        initial_cost_value = "NULL" if record['initial_cost'] is None else f"{record['initial_cost']}"
        installation_cost_value = "NULL" if record['installation_cost'] is None else f"{record['installation_cost']}"
        annual_maintenance_value = "NULL" if record['annual_maintenance_cost'] is None else f"{record['annual_maintenance_cost']}"
        replacement_cost_value = "NULL" if record['replacement_cost'] is None else f"{record['replacement_cost']}"
        annual_energy_value = "NULL" if record['annual_energy_cost'] is None else f"{record['annual_energy_cost']}"
        cost_date_value = "NULL" if record['cost_date'] is None else f"'{record['cost_date']}'"
        comments_value = "NULL" if record['cost_comments'] is None else f"'{record['cost_comments'].replace('\'', '\'\'')}'"
        
        if cost:
            # Update existing cost record
            update_query = f"""
                UPDATE Equipment_Costs
                SET 
                    InitialCost = {initial_cost_value},
                    InstallationCost = {installation_cost_value},
                    AnnualMaintenanceCost = {annual_maintenance_value},
                    ReplacementCost = {replacement_cost_value},
                    AnnualEnergyCost = {annual_energy_value},
                    CostDate = {cost_date_value},
                    Comments = {comments_value}
                WHERE CostID = {cost['CostID']}
            """
            conn.execute(text(update_query))
        else:
            # Create new cost record
            insert_query = f"""
                INSERT INTO Equipment_Costs(
                    EquipmentID,
                    InitialCost,
                    InstallationCost,
                    AnnualMaintenanceCost,
                    ReplacementCost,
                    AnnualEnergyCost,
                    CostDate,
                    Comments
                )
                VALUES (
                    {equipment_id},
                    {initial_cost_value},
                    {installation_cost_value},
                    {annual_maintenance_value},
                    {replacement_cost_value},
                    {annual_energy_value},
                    {cost_date_value},
                    {comments_value}
                )
            """
            conn.execute(text(insert_query))
            
        conn.commit()
    except Exception as e:
        logger.error(f"Error processing costs: {str(e)}")
        raise Exception(f"Error processing costs: {str(e)}")