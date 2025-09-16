import csv
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


class LicensePlateDatabase:
    """Simple license plate database for validation."""

    def __init__(self, csv_file: str = "mock_license_plates.csv"):
        self.csv_file = csv_file
        self.database = self.load_database()

    def load_database(self) -> List[Dict]:
        """Load license plate database from CSV file."""
        database = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                database = list(reader)
            print(f"Loaded {len(database)} license plates from database")
        except FileNotFoundError:
            print(f"Warning: {self.csv_file} not found. Creating empty database.")
        return database

    def normalize_plate(self, plate_text: str) -> str:
        """Normalize license plate text for comparison."""
        return re.sub(r'\s+', ' ', plate_text.strip().upper())

    def exact_match(self, detected_plate: str) -> Optional[Dict]:
        """Check for exact match in database."""
        normalized_detected = self.normalize_plate(detected_plate)

        for record in self.database:
            normalized_db = self.normalize_plate(record['plate_number'])
            if normalized_detected == normalized_db:
                return record
        return None

    def fuzzy_match(self, detected_plate: str, threshold: float = 0.8) -> Optional[Tuple[Dict, float]]:
        """Check for fuzzy match in database."""
        normalized_detected = self.normalize_plate(detected_plate)
        best_match = None
        best_score = 0.0

        for record in self.database:
            normalized_db = self.normalize_plate(record['plate_number'])
            similarity = SequenceMatcher(None, normalized_detected, normalized_db).ratio()

            if similarity >= threshold and similarity > best_score:
                best_score = similarity
                best_match = record

        if best_match:
            return best_match, best_score
        return None

    def validate_license_plate(self, detected_plate: str, confidence: float,
                             confidence_threshold: float = 0.2) -> Dict:
        """Comprehensive license plate validation."""
        result = {
            'detected_plate': detected_plate,
            'confidence': confidence,
            'validation_status': 'FAIL',
            'validation_reason': '',
            'database_match': None,
            'match_type': None,
            'match_score': 0.0
        }

        # Step 1: Check confidence threshold
        if confidence < confidence_threshold:
            result['validation_reason'] = f"Low confidence: {confidence:.3f} < {confidence_threshold}"
            return result

        # Step 2: Check for exact match
        exact_match = self.exact_match(detected_plate)
        if exact_match:
            result.update({
                'validation_status': 'PASS',
                'validation_reason': 'Exact match in database',
                'database_match': exact_match,
                'match_type': 'exact',
                'match_score': 1.0
            })
            return result

        # Step 3: Check for fuzzy match
        fuzzy_result = self.fuzzy_match(detected_plate, threshold=0.8)
        if fuzzy_result:
            match_record, score = fuzzy_result
            result.update({
                'validation_status': 'PASS',
                'validation_reason': f'Fuzzy match in database (similarity: {score:.2f})',
                'database_match': match_record,
                'match_type': 'fuzzy',
                'match_score': score
            })
            return result

        # Step 4: No match found
        result['validation_reason'] = 'No match found in database'
        return result

    def add_license_plate(self, plate_number: str, province: str,
                         vehicle_type: str = 'private', status: str = 'active'):
        """Add new license plate to database."""
        new_record = {
            'plate_number': plate_number,
            'province': province,
            'vehicle_type': vehicle_type,
            'status': status
        }
        self.database.append(new_record)

        # Save to CSV
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.database:
                fieldnames = self.database[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.database)

        print(f"Added {plate_number} to database")

    def search_by_province(self, province: str) -> List[Dict]:
        """Search license plates by province."""
        matches = []
        for record in self.database:
            if province.lower() in record['province'].lower():
                matches.append(record)
        return matches

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self.database:
            return {'total': 0, 'by_province': {}, 'by_vehicle_type': {}, 'by_status': {}}

        stats = {
            'total': len(self.database),
            'by_province': {},
            'by_vehicle_type': {},
            'by_status': {}
        }

        for record in self.database:
            province = record.get('province', 'unknown')
            vehicle_type = record.get('vehicle_type', 'unknown')
            status = record.get('status', 'unknown')

            stats['by_province'][province] = stats['by_province'].get(province, 0) + 1
            stats['by_vehicle_type'][vehicle_type] = stats['by_vehicle_type'].get(vehicle_type, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

        return stats


def demo_validation():
    """Demonstrate license plate validation."""
    db = LicensePlateDatabase()

    # Test cases
    test_cases = [
        ("กก 555", 0.95),      # Exact match
        ("กก555", 0.90),       # Similar format
        ("ซค 5", 0.88),        # Exact match
        ("กท 2058", 0.92),     # Exact match
        ("กข 999", 0.85),      # Not in database
        ("abc 123", 0.70),     # Low confidence, not Thai
    ]

    print("LICENSE PLATE VALIDATION DEMO")
    print("=" * 50)

    for plate, confidence in test_cases:
        result = db.validate_license_plate(plate, confidence)

        print(f"\nPlate: {plate}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Status: {result['validation_status']}")
        print(f"Reason: {result['validation_reason']}")

        if result['database_match']:
            match = result['database_match']
            print(f"Match: {match['plate_number']} ({match['province']}, {match['vehicle_type']})")

    # Show statistics
    print(f"\nDATABASE STATISTICS")
    print("=" * 30)
    stats = db.get_statistics()
    print(f"Total plates: {stats['total']}")
    print(f"Provinces: {list(stats['by_province'].keys())}")
    print(f"Vehicle types: {list(stats['by_vehicle_type'].keys())}")


if __name__ == "__main__":
    demo_validation()