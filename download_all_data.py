#!/usr/bin/env python3
"""
Master Data Download Script

Downloads all data sources in prioritized order:
1. GitHub repositories (easiest, fastest)
2. Kaggle datasets (medium, requires credentials)
3. HuggingFace datasets (large, time-consuming)

This script orchestrates the entire data collection process.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add crawlers to path
sys.path.insert(0, str(Path(__file__).parent / "crawlers"))

from config.settings import OUTPUT_DIR
from utils.logger import log


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def save_master_summary(summary: dict):
    """Save master download summary to file."""
    summary_file = OUTPUT_DIR / "master_download_summary.json"
    summary["timestamp"] = datetime.now().isoformat()
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.success(f"Master summary saved to: {summary_file}")


def download_github_repos(priority: str = "high", dry_run: bool = False):
    """Download GitHub repositories."""
    from cloners.clone_all import clone_all_repos
    from sources.source_types import Priority

    print_section_header("PHASE 1: GitHub Repositories")
    log.info(f"Downloading {priority.upper()} priority GitHub repositories")

    if dry_run:
        log.info("DRY RUN MODE - No actual downloads will occur")

    priority_enum = Priority(priority.upper()) if priority else None

    results, summary = clone_all_repos(
        categories=None,  # All categories
        priority_filter=priority_enum,
        dry_run=dry_run
    )

    return {
        "phase": "github",
        "priority": priority,
        "summary": summary,
        "success": summary["failed"] == 0 if summary else False
    }


def download_kaggle_datasets(force: bool = False):
    """Download Kaggle datasets."""
    from downloaders.kaggle_downloader import KaggleDownloader

    print_section_header("PHASE 2: Kaggle Datasets")
    log.info("Downloading Kaggle datasets")

    downloader = KaggleDownloader()
    results = downloader.download_all_defaults(force=force)

    successful = sum(1 for path in results.values() if path is not None)
    failed = sum(1 for path in results.values() if path is None)

    summary = {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "datasets": {k: str(v) if v else None for k, v in results.items()}
    }

    return {
        "phase": "kaggle",
        "summary": summary,
        "success": failed == 0
    }


def download_huggingface_datasets(force: bool = False, streaming: bool = False):
    """Download HuggingFace datasets."""
    from downloaders.hf_downloader import HuggingFaceDownloader

    print_section_header("PHASE 3: HuggingFace Datasets")
    log.info("Downloading HuggingFace datasets")

    if not streaming:
        log.warning("WARNING: The Zellic dataset is VERY LARGE (~50GB+)")
        log.warning("This will take a significant amount of time...")
        response = input("Continue with full download? (y/N): ").strip().lower()
        if response != 'y':
            log.info("Skipping HuggingFace download")
            return {
                "phase": "huggingface",
                "summary": {"skipped": True},
                "success": True
            }

    downloader = HuggingFaceDownloader()
    results = downloader.download_all_defaults(force=force)

    successful = sum(1 for path in results.values() if path is not None)
    failed = sum(1 for path in results.values() if path is None)

    summary = {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "datasets": {k: str(v) if v else None for k, v in results.items()}
    }

    return {
        "phase": "huggingface",
        "summary": summary,
        "success": failed == 0
    }


def main():
    """Main orchestration function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download all smart contract data sources in prioritized order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script downloads data in the following order:
  1. GitHub repositories (HIGH priority only by default)
  2. Kaggle datasets (requires credentials)
  3. HuggingFace datasets (very large, optional)

SETUP REQUIREMENTS:
  1. GitHub: Set GITHUB_TOKEN environment variable (optional but recommended)
  2. Kaggle: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
  3. HuggingFace: Set HUGGINGFACE_TOKEN environment variable (optional)

Run verify_setup.py first to check your configuration.

Examples:
  # Download everything with default settings
  python download_all_data.py

  # Download only HIGH priority GitHub repos + Kaggle
  python download_all_data.py --skip-huggingface

  # Include MEDIUM priority GitHub repos
  python download_all_data.py --github-priority medium

  # Dry run to see what would be downloaded
  python download_all_data.py --dry-run

  # Force re-download everything
  python download_all_data.py --force
        """
    )

    parser.add_argument(
        "--github-priority",
        choices=["high", "medium", "low"],
        default="high",
        help="GitHub repository priority filter (default: high)"
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub repository downloads"
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle dataset downloads"
    )
    parser.add_argument(
        "--skip-huggingface",
        action="store_true",
        help="Skip HuggingFace dataset downloads"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )

    args = parser.parse_args()

    print_section_header("SMART CONTRACT DATA DOWNLOAD - MASTER SCRIPT")

    log.info("Starting complete data download process")
    log.info(f"Output directory: {OUTPUT_DIR}")

    master_summary = {
        "phases": [],
        "overall_success": True
    }

    # Phase 1: GitHub repositories
    if not args.skip_github:
        try:
            github_result = download_github_repos(
                priority=args.github_priority,
                dry_run=args.dry_run
            )
            master_summary["phases"].append(github_result)
            if not github_result["success"]:
                master_summary["overall_success"] = False
                log.warning("Some GitHub repositories failed to download")
        except Exception as e:
            log.error(f"GitHub download phase failed: {e}")
            master_summary["overall_success"] = False
    else:
        log.info("Skipping GitHub repositories (--skip-github)")

    # Phase 2: Kaggle datasets
    if not args.skip_kaggle and not args.dry_run:
        try:
            kaggle_result = download_kaggle_datasets(force=args.force)
            master_summary["phases"].append(kaggle_result)
            if not kaggle_result["success"]:
                master_summary["overall_success"] = False
                log.warning("Some Kaggle datasets failed to download")
        except Exception as e:
            log.error(f"Kaggle download phase failed: {e}")
            log.info("Make sure KAGGLE_USERNAME and KAGGLE_KEY are set")
            master_summary["overall_success"] = False
    elif args.dry_run:
        log.info("Skipping Kaggle datasets (dry run mode)")
    else:
        log.info("Skipping Kaggle datasets (--skip-kaggle)")

    # Phase 3: HuggingFace datasets
    if not args.skip_huggingface and not args.dry_run:
        try:
            hf_result = download_huggingface_datasets(force=args.force)
            master_summary["phases"].append(hf_result)
            if not hf_result["success"]:
                master_summary["overall_success"] = False
                log.warning("Some HuggingFace datasets failed to download")
        except Exception as e:
            log.error(f"HuggingFace download phase failed: {e}")
            master_summary["overall_success"] = False
    elif args.dry_run:
        log.info("Skipping HuggingFace datasets (dry run mode)")
    else:
        log.info("Skipping HuggingFace datasets (--skip-huggingface)")

    # Final summary
    print_section_header("DOWNLOAD COMPLETE - SUMMARY")

    for phase in master_summary["phases"]:
        print(f"\n{phase['phase'].upper()}:")
        if "skipped" in phase["summary"]:
            print("  Status: SKIPPED")
        else:
            summary = phase["summary"]
            if "total" in summary:
                print(f"  Total: {summary['total']}")
                print(f"  Successful: {summary.get('successful', 0)}")
                print(f"  Failed: {summary.get('failed', 0)}")
            else:
                print(f"  Cloned: {summary.get('cloned', 0)}")
                print(f"  Updated: {summary.get('updated', 0)}")
                print(f"  Failed: {summary.get('failed', 0)}")

    overall_status = "✓ SUCCESS" if master_summary["overall_success"] else "✗ PARTIAL SUCCESS"
    print(f"\nOverall Status: {overall_status}\n")

    # Save summary
    if not args.dry_run:
        save_master_summary(master_summary)

    print("="*70 + "\n")

    # Exit code
    sys.exit(0 if master_summary["overall_success"] else 1)


if __name__ == "__main__":
    main()
