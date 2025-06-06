#!/usr/bin/env python3
"""
TrustRAG Duplicate Cleaner
Removes 'both_sides' variations when 'both_combined' versions exist
"""

import os
import re
from pathlib import Path
from collections import defaultdict


class TrustRAGDuplicateCleaner:
    def __init__(self, logs_directory, dry_run=True):
        self.logs_dir = Path(logs_directory)
        self.dry_run = dry_run
        self.deleted_files = []
        self.processed_folders = []

    def get_experiment_base_name(self, filename):
        """Extract the base experiment name without the both_sides/both_combined suffix"""
        # Remove .log extension
        name = filename.replace(".log", "")

        # Remove the both_sides or both_combined suffix
        if name.endswith("_no_questions_both_sides"):
            base_name = name.replace("_no_questions_both_sides", "")
        elif name.endswith("_both_combined"):
            base_name = name.replace("_both_combined", "")
        else:
            # If it doesn't end with either, return the full name
            base_name = name

        return base_name

    def find_duplicates_in_folder(self, folder_path):
        """Find pairs where both_sides and both_combined versions exist"""
        log_files = list(folder_path.glob("*.log"))

        # Group files by their base experiment name
        experiment_groups = defaultdict(list)

        for log_file in log_files:
            base_name = self.get_experiment_base_name(log_file.name)
            experiment_groups[base_name].append(log_file)

        # Find groups that have both versions
        duplicates_to_remove = []

        for base_name, files in experiment_groups.items():
            if len(files) == 2:
                # Check if we have both versions
                both_sides_file = None
                both_combined_file = None

                for file in files:
                    if file.name.endswith("_no_questions_both_sides.log"):
                        both_sides_file = file
                    elif file.name.endswith("_both_combined.log"):
                        both_combined_file = file

                # If we have both, mark both_sides for deletion
                if both_sides_file and both_combined_file:
                    duplicates_to_remove.append(
                        {
                            "base_name": base_name,
                            "to_delete": both_sides_file,
                            "to_keep": both_combined_file,
                        }
                    )

        return duplicates_to_remove

    def clean_folder(self, folder_path):
        """Clean duplicates in a single folder"""
        print(f"\n📁 Processing folder: {folder_path.name}")

        duplicates = self.find_duplicates_in_folder(folder_path)

        if not duplicates:
            print("   ✅ No duplicates found")
            return []

        deleted_in_folder = []

        for duplicate in duplicates:
            file_to_delete = duplicate["to_delete"]
            file_to_keep = duplicate["to_keep"]

            print(f"   🔍 Found duplicate pair:")
            print(f"      🗑️  DELETE: {file_to_delete.name}")
            print(f"      ✅ KEEP:   {file_to_keep.name}")

            if not self.dry_run:
                try:
                    file_to_delete.unlink()  # Delete the file
                    print(f"      ✅ Successfully deleted: {file_to_delete.name}")
                    deleted_in_folder.append(str(file_to_delete))
                except Exception as e:
                    print(f"      ❌ Error deleting {file_to_delete.name}: {e}")
            else:
                print(f"      📝 DRY RUN - Would delete: {file_to_delete.name}")
                deleted_in_folder.append(str(file_to_delete))

        return deleted_in_folder

    def clean_all_folders(self):
        """Clean duplicates in all experiment folders"""
        print("🧹 TrustRAG Duplicate Cleaner")
        print("=" * 50)

        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will actually be deleted")
        else:
            print("⚠️  LIVE MODE - Files will be permanently deleted!")

        print(
            "📋 Looking for experiments where both 'both_sides' and 'both_combined' exist"
        )
        print("🎯 Strategy: Keep 'both_combined', delete 'both_sides'")

        # Find all experiment folders
        folders = [
            item
            for item in self.logs_dir.iterdir()
            if item.is_dir() and item.name.startswith("dataset_")
        ]

        print(f"\n📂 Found {len(folders)} experiment folders to process")

        total_deleted = []

        for folder in folders:
            deleted_in_folder = self.clean_folder(folder)
            total_deleted.extend(deleted_in_folder)
            self.processed_folders.append(folder.name)

        # Summary
        print("\n" + "=" * 50)
        print("📊 CLEANUP SUMMARY")
        print("=" * 50)

        if total_deleted:
            print(
                f"🗑️  Total files {'marked for deletion' if self.dry_run else 'deleted'}: {len(total_deleted)}"
            )
            print("\n📝 List of deleted files:")
            for deleted_file in total_deleted:
                print(f"   • {deleted_file}")
        else:
            print("✅ No duplicate files found - nothing to clean!")

        print(f"\n📁 Processed {len(self.processed_folders)} folders")

        if self.dry_run:
            print("\n💡 To actually delete files, run with dry_run=False")

        self.deleted_files = total_deleted
        return total_deleted

    def generate_report(self, output_file="cleanup_report.txt"):
        """Generate a detailed report of the cleanup operation"""
        with open(output_file, "w") as f:
            f.write("TrustRAG Duplicate Cleanup Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}\n")
            f.write(f"Processed folders: {len(self.processed_folders)}\n")
            f.write(f"Files deleted: {len(self.deleted_files)}\n\n")

            if self.deleted_files:
                f.write("Deleted files:\n")
                for deleted_file in self.deleted_files:
                    f.write(f"  - {deleted_file}\n")
            else:
                f.write("No files were deleted.\n")

        print(f"📄 Report saved to: {output_file}")


# Usage
if __name__ == "__main__":
    # Initialize cleaner in DRY RUN mode first (safe)
    cleaner = TrustRAGDuplicateCleaner("logs", dry_run=False)

    print("🔍 STEP 1: DRY RUN - Checking what would be deleted")
    deleted_files = cleaner.clean_all_folders()

    if deleted_files:
        print("\n" + "⚠️ " * 20)
        print("⚠️  Ready to delete files? Change dry_run=False to proceed")
        print("⚠️ " * 20)

    # Generate report
    cleaner.generate_report()
