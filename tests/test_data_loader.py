import unittest
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import DataLoader, DataSource, SolidityContract

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # We assume the 'output' directory exists as it was verified in previous steps
        self.loader = DataLoader(data_dir="output")

    def test_list_available_sources(self):
        sources = self.loader.list_available_sources()
        print(f"Available sources: {sources}")
        self.assertIsInstance(sources, list)
        # Check for at least one expected source
        self.assertTrue(any(s in sources for s in ["smartbugs", "kaggle", "audit_reports"]))

    def test_load_smartbugs(self):
        contracts = list(self.loader.load_smartbugs())
        print(f"Loaded {len(contracts)} SmartBugs contracts")
        if len(contracts) > 0:
            contract = contracts[0]
            self.assertIsInstance(contract, SolidityContract)
            self.assertTrue(contract.is_vulnerable)
            self.assertIn("smartbugs", contract.metadata["source"])

    def test_load_kaggle(self):
        contracts = list(self.loader.load_kaggle_vulnerability())
        print(f"Loaded {len(contracts)} Kaggle contracts")
        if len(contracts) > 0:
            contract = contracts[0]
            self.assertIsInstance(contract, SolidityContract)
            self.assertTrue(contract.is_vulnerable)
            self.assertTrue(contract.metadata["source"].startswith("kaggle"))

    def test_load_audit_reports(self):
        reports = list(self.loader.load_audit_reports())
        print(f"Loaded {len(reports)} audit reports")
        if len(reports) > 0:
            report = reports[0]
            self.assertIn("content", report)
            self.assertIn("source", report)
            self.assertEqual(report["type"], "audit_report")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

