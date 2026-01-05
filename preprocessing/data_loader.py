"""
Data Loader Module
==================
Loads raw data from various sources and formats for training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources as per PRD Section 5.1"""
    SMARTBUGS_CURATED = "smartbugs_curated"       # 143 contracts, 208 vulns
    KAGGLE_VULNERABILITY = "kaggle_vulnerability"  # 12K+ contracts
    ZELLIC_FIESTA = "zellic_fiesta"               # 514K deduplicated
    CODE4RENA = "code4rena"                        # Audit reports
    SHERLOCK = "sherlock"                          # Audit reports
    DEFIHACKLABS = "defihacklabs"                 # 550+ incidents
    ETHERNAUT = "ethernaut"                        # Educational
    CYFRIN = "cyfrin"                              # Educational
    RARESKILLS = "rareskills"                      # Educational


@dataclass
class SolidityContract:
    """Represents a Solidity smart contract"""
    source_code: str
    file_path: str
    contract_name: Optional[str] = None
    compiler_version: Optional[str] = None
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    is_vulnerable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VulnerabilityLabel:
    """Represents a vulnerability annotation"""
    swc_id: str
    cwe_id: Optional[str] = None
    name: str = ""
    severity: str = "Medium"
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    description: Optional[str] = None
    remediation: Optional[str] = None


class DataLoader:
    """
    Unified data loader for all data sources.

    Expects data in /data directory with structure:
    /data
    ├── smartbugs/           # SmartBugs-curated dataset
    ├── kaggle/              # Kaggle vulnerability dataset
    ├── zellic/              # Zellic smart-contract-fiesta
    ├── audit_reports/       # Code4rena, Sherlock, etc.
    ├── exploits/            # DeFiHackLabs
    └── educational/         # Ethernaut, Cyfrin, RareSkills
    """

    def __init__(self, data_dir: Union[str, Path] = "output"):
        self.data_dir = Path(data_dir)
        self._validate_data_dir()

    def _validate_data_dir(self):
        """Validate data directory exists"""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. "
                          "Please ensure data is available before training.")

    def list_available_sources(self) -> List[str]:
        """List available data sources in the data directory"""
        if not self.data_dir.exists():
            return []
        sources = []
        if (self.data_dir / "repos/vulnerability_datasets/smartbugs-curated").exists():
            sources.append("smartbugs")
        if (self.data_dir / "datasets/kaggle").exists():
            sources.append("kaggle")
        if (self.data_dir / "datasets/huggingface/Zellic_smart-contract-fiesta").exists():
            sources.append("zellic")
        if (self.data_dir / "repos/audit_repos").exists():
            sources.append("audit_reports")
        return sources

    def load_smartbugs(self) -> Iterator[SolidityContract]:
        """
        Load SmartBugs-curated dataset.
        Expected structure:
        output/repos/vulnerability_datasets/smartbugs-curated/
        ├── dataset/
        │   └── <vulnerability_type>/
        │       └── <contract>.sol
        └── vulnerabilities.json
        """
        smartbugs_dir = self.data_dir / "repos/vulnerability_datasets/smartbugs-curated"
        if not smartbugs_dir.exists():
            logger.warning(f"SmartBugs directory not found at {smartbugs_dir}")
            return

        # Load vulnerability annotations if available
        vuln_file = smartbugs_dir / "vulnerabilities.json"
        vuln_map = {}
        if vuln_file.exists():
            with open(vuln_file) as f:
                vuln_map = json.load(f)

        # Load contracts
        dataset_dir = smartbugs_dir / "dataset"
        if dataset_dir.exists():
            for vuln_type_dir in dataset_dir.iterdir():
                if vuln_type_dir.is_dir():
                    vuln_type = vuln_type_dir.name
                    for sol_file in vuln_type_dir.glob("*.sol"):
                        with open(sol_file) as f:
                            source_code = f.read()

                        contract = SolidityContract(
                            source_code=source_code,
                            file_path=str(sol_file),
                            is_vulnerable=True,
                            vulnerabilities=[{
                                "type": vuln_type,
                                "swc_id": self._get_swc_for_type(vuln_type),
                            }],
                            metadata={"source": "smartbugs", "category": vuln_type}
                        )
                        yield contract

    def load_kaggle_vulnerability(self) -> Iterator[SolidityContract]:
        """
        Load Kaggle smart contract vulnerability datasets.
        Expected structure:
        output/datasets/kaggle/
        ├── tranduongminhdai_smart-contract-vulnerability-datset/
        └── bcccdatasets_bccc-vulscs-2023/
        """
        kaggle_dir = self.data_dir / "datasets/kaggle"
        if not kaggle_dir.exists():
            logger.warning("Kaggle directory not found")
            return

        # 1. Load tranduongminhdai dataset
        tdm_dir = kaggle_dir / "tranduongminhdai_smart-contract-vulnerability-datset"
        if tdm_dir.exists():
            csv_file = tdm_dir / "SC_Vuln_8label.csv"
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    contract = SolidityContract(
                        source_code=row.get("source_code", row.get("code", "")),
                        file_path="tranduongminhdai/" + str(row.get("id", "unknown")),
                        is_vulnerable=True, # This is the vuln dataset
                        metadata={"source": "kaggle_tdm"}
                    )
                    yield contract

        # 2. Load BCCC dataset
        bccc_dir = kaggle_dir / "bcccdatasets_bccc-vulscs-2023"
        if bccc_dir.exists():
            vuln_csv = bccc_dir / "BCCC-VolSCs-2023_Vulnerable.csv"
            if vuln_csv.exists():
                import pandas as pd
                df = pd.read_csv(vuln_csv)
                for _, row in df.iterrows():
                    # Look for code in Vulnerable_SourceCodes folder
                    code_path = bccc_dir / "Vulnerable_SourceCodes" / f"{row.get('Contract Address', row.get('Address'))}.sol"
                    source_code = ""
                    if code_path.exists():
                        with open(code_path) as f:
                            source_code = f.read()
                    
                    contract = SolidityContract(
                        source_code=source_code or row.get("Source Code", ""),
                        file_path=str(code_path),
                        is_vulnerable=True,
                        metadata={"source": "kaggle_bccc_vuln"}
                    )
                    yield contract

    def load_zellic(self, streaming: bool = True) -> Iterator[SolidityContract]:
        """
        Load Zellic smart-contract-fiesta dataset.
        Expected structure:
        output/datasets/huggingface/Zellic_smart-contract-fiesta/
        """
        zellic_dir = self.data_dir / "datasets/huggingface/Zellic_smart-contract-fiesta"
        if not zellic_dir.exists():
            logger.warning(f"Zellic directory not found at {zellic_dir}")
            return

        # Try parquet format (HuggingFace download)
        parquet_files = list(zellic_dir.glob("*.parquet"))
        if parquet_files:
            import pyarrow.parquet as pq
            for pq_file in parquet_files:
                table = pq.read_table(pq_file)
                for batch in table.to_batches():
                    for i in range(batch.num_rows):
                        row = {col: batch.column(col)[i].as_py()
                               for col in batch.schema.names}
                        contract = SolidityContract(
                            source_code=row.get("source_code", row.get("content", "")),
                            file_path=row.get("file_path", "unknown"),
                            is_vulnerable=False,  # Zellic is unlabeled
                            metadata={"source": "zellic"}
                        )
                        yield contract
            return

        # Try directory of .sol files
        contracts_dir = zellic_dir / "contracts"
        if contracts_dir.exists():
            for sol_file in contracts_dir.rglob("*.sol"):
                with open(sol_file) as f:
                    source_code = f.read()
                contract = SolidityContract(
                    source_code=source_code,
                    file_path=str(sol_file),
                    is_vulnerable=False,
                    metadata={"source": "zellic"}
                )
                yield contract

    def load_defihacklabs(self) -> Iterator[Dict[str, Any]]:
        """
        Load DeFiHackLabs exploit database.
        Expected structure:
        /data/exploits/defihacklabs/
        └── src/
            └── test/
                └── <year>/
                    └── <exploit>.sol
        """
        defihack_dir = self.data_dir / "exploits" / "defihacklabs"
        if not defihack_dir.exists():
            defihack_dir = self.data_dir / "defihacklabs"

        if not defihack_dir.exists():
            logger.warning("DeFiHackLabs directory not found")
            return

        src_dir = defihack_dir / "src" / "test"
        if not src_dir.exists():
            src_dir = defihack_dir

        for sol_file in src_dir.rglob("*.sol"):
            with open(sol_file) as f:
                source_code = f.read()

            # Extract year and exploit name from path
            parts = sol_file.parts
            year = None
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = part
                    break

            yield {
                "source_code": source_code,
                "file_path": str(sol_file),
                "exploit_name": sol_file.stem,
                "year": year,
                "metadata": {"source": "defihacklabs"}
            }

    def load_audit_reports(self) -> Iterator[Dict[str, Any]]:
        """
        Load audit reports from output/repos/audit_repos/.
        """
        audit_dir = self.data_dir / "repos/audit_repos"
        if not audit_dir.exists():
            logger.warning("Audit reports directory not found")
            return

        for source_dir in audit_dir.iterdir():
            if source_dir.is_dir():
                source_name = source_dir.name

                # Look for markdown or JSON files
                for report_file in source_dir.rglob("*.md"):
                    with open(report_file) as f:
                        content = f.read()
                    yield {
                        "content": content,
                        "file_path": str(report_file),
                        "source": source_name,
                        "type": "audit_report"
                    }

                for report_file in source_dir.rglob("*.json"):
                    with open(report_file) as f:
                        data = json.load(f)
                    yield {
                        "content": data,
                        "file_path": str(report_file),
                        "source": source_name,
                        "type": "audit_report"
                    }

    def load_all_contracts(
        self,
        sources: Optional[List[DataSource]] = None,
        limit: Optional[int] = None
    ) -> Iterator[SolidityContract]:
        """
        Load contracts from multiple sources.

        Args:
            sources: List of data sources to load. If None, loads all available.
            limit: Maximum number of contracts to load (for testing).
        """
        if sources is None:
            sources = [
                DataSource.SMARTBUGS_CURATED,
                DataSource.KAGGLE_VULNERABILITY,
                DataSource.ZELLIC_FIESTA,
            ]

        count = 0
        for source in sources:
            if limit and count >= limit:
                break

            loader_map = {
                DataSource.SMARTBUGS_CURATED: self.load_smartbugs,
                DataSource.KAGGLE_VULNERABILITY: self.load_kaggle_vulnerability,
                DataSource.ZELLIC_FIESTA: self.load_zellic,
            }

            loader = loader_map.get(source)
            if loader:
                for contract in loader():
                    if limit and count >= limit:
                        break
                    yield contract
                    count += 1

    @staticmethod
    def _get_swc_for_type(vuln_type: str) -> str:
        """Map vulnerability type names to SWC IDs"""
        mapping = {
            "reentrancy": "SWC-107",
            "integer_overflow": "SWC-101",
            "integer_underflow": "SWC-101",
            "access_control": "SWC-115",
            "unchecked_call": "SWC-104",
            "unchecked_low_level_calls": "SWC-104",
            "front_running": "SWC-114",
            "time_manipulation": "SWC-116",
            "bad_randomness": "SWC-120",
            "denial_of_service": "SWC-128",
            "short_addresses": "SWC-129",
            "tx_origin": "SWC-115",
        }
        return mapping.get(vuln_type.lower(), "SWC-000")


def get_data_stats(data_dir: Union[str, Path] = "/data") -> Dict[str, Any]:
    """Get statistics about available data"""
    loader = DataLoader(data_dir)

    stats = {
        "available_sources": loader.list_available_sources(),
        "counts": {}
    }

    # Count contracts per source (sample for large datasets)
    for source in [DataSource.SMARTBUGS_CURATED, DataSource.KAGGLE_VULNERABILITY]:
        count = 0
        try:
            loader_map = {
                DataSource.SMARTBUGS_CURATED: loader.load_smartbugs,
                DataSource.KAGGLE_VULNERABILITY: loader.load_kaggle_vulnerability,
            }
            for _ in loader_map[source]():
                count += 1
        except Exception as e:
            logger.warning(f"Error counting {source.value}: {e}")
        stats["counts"][source.value] = count

    return stats
