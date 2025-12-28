#!/usr/bin/env python3
"""
Database Export Script for Telegram Bot
---------------------------------------
This script exports your PostgreSQL database data to SQL statements 
that can be imported into another database, making migration easier.
"""

import os
import datetime
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_database():
    """Export the PostgreSQL database to a SQL file."""
    try:
        # Get database connection info from environment variables
        db_url = os.environ.get("DATABASE_URL")
        
        if not db_url:
            logger.error("DATABASE_URL environment variable not found")
            return False
        
        # Extract database connection details
        # Format: postgres://username:password@hostname:port/database_name
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        # Create a timestamp for the backup file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"database_backup_{timestamp}.sql"
        
        # Create the pg_dump command
        dump_command = f"pg_dump {db_url} --no-owner --no-acl > {backup_filename}"
        
        # Execute the command
        logger.info(f"Starting database export to {backup_filename}")
        subprocess.run(dump_command, shell=True, check=True)
        
        logger.info(f"Database successfully exported to {backup_filename}")
        logger.info(f"File location: {os.path.abspath(backup_filename)}")
        
        # Print instructions for importing
        print("\n\n" + "="*80)
        print("DATABASE EXPORT SUCCESSFUL")
        print("="*80)
        print(f"\nThe database has been exported to: {os.path.abspath(backup_filename)}")
        print("\nTo import this database on your deployment platform:")
        
        print("\n1. For Heroku:")
        print("   heroku pg:psql --app your-app-name < " + backup_filename)
        
        print("\n2. For Render:")
        print("   psql YOUR_RENDER_DATABASE_URL < " + backup_filename)
        
        print("\n3. For local testing:")
        print("   psql -U username -d database_name < " + backup_filename)
        
        print("\nMake sure you have the PostgreSQL client tools installed")
        print("and that you've authenticated with your deployment platform.")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting database: {e}")
        return False

if __name__ == "__main__":
    export_database()