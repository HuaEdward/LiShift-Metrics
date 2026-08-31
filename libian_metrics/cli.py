"""
Command-line interface for LiBian metrics computation.
"""

import argparse
import json
import os
import sys
import logging

import pandas as pd

from .io_utils import read_image
from .preprocess import preprocess
from .metrics import compute_all_metrics
from .calibrate import load_calibration, _get_default_calibration
from .visualize import save_metric_visualization
from .batch_process import process_dataset_folder, print_results
from .compare import (
    compare_result_files,
    print_comparison,
    save_comparison_csv,
    save_comparison_json,
    save_comparison_visualizations,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LiBian Metrics: Ancient character glyph quality assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python -m libian_metrics --image glyph.jpg
    python -m libian_metrics --image glyph.jpg --calib calibration.json --out result.json
  
  Batch processing:
    python -m libian_metrics --dataset data/my_dataset --out results/output.json
    python -m libian_metrics --dataset data/my_dataset --calib calib.json --detailed

  Paired comparison (after - before):
    python -m libian_metrics --compare before.json after.json --out comparison.json
    python -m libian_metrics --compare before.json after.json --csv comparison.csv --visualize
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', default=None, type=str, help='Path to input image (JPG/PNG)')
    input_group.add_argument('--dataset', default=None, type=str, help='Path to dataset folder')
    input_group.add_argument(
        '--compare',
        nargs=2,
        metavar=('BEFORE_JSON', 'AFTER_JSON'),
        help='Compare two batch result JSON files using paired characters',
    )
    parser.add_argument('--calib', default=None, type=str, help='Path to calibration JSON')
    parser.add_argument('--out', default=None, type=str, help='Output JSON file path')
    parser.add_argument('--csv', default=None, type=str, help='Output CSV file path')
    parser.add_argument(
        '--min-samples',
        default=1,
        type=int,
        help='Minimum images per dataset for each character in comparison mode',
    )
    parser.add_argument('--detailed', action='store_true', help='Show detailed metrics for batch')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--viz-dir', default='runs', type=str, help='Visualizations directory')
    
    args = parser.parse_args()
    
    if args.compare:
        _process_comparison(args)
    elif args.dataset:
        _process_batch(args)
    else:
        _process_single_image(args)


def _process_comparison(args):
    """Compare two previously generated batch result files."""
    try:
        before_path, after_path = args.compare
        report = compare_result_files(before_path, after_path, min_samples=args.min_samples)
        print_comparison(report, detailed=args.detailed)

        if args.out:
            output_path = save_comparison_json(report, args.out)
            print(f"Comparison JSON saved to {output_path}")
        if args.csv:
            csv_path = save_comparison_csv(report, args.csv)
            print(f"Comparison CSV saved to {csv_path}")
        if args.visualize:
            paths = save_comparison_visualizations(report, args.viz_dir)
            print("Comparison visualizations saved:")
            for path in paths:
                print(f"  {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _process_batch(args):
    """Process dataset folder."""
    try:
        calib = None
        if args.calib:
            calib = load_calibration(args.calib)
        else:
            calib = _get_default_calibration()
        
        results = process_dataset_folder(args.dataset, calib=calib, output_json=args.out)
        print_results(results, detailed=args.detailed)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _process_single_image(args):
    """Process single image file."""
    try:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        
        img = read_image(args.image)
        bin_img, skel, meta = preprocess(img)
        
        calib = None
        if args.calib:
            calib = load_calibration(args.calib)
        else:
            calib = _get_default_calibration()
        
        metrics = compute_all_metrics(bin_img, skel, calib)
        payload = {'image': args.image, **metrics, **meta}
        
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {args.out}")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        
        if args.csv:
            df = pd.DataFrame([payload])
            file_exists = os.path.exists(args.csv)
            if file_exists:
                df.to_csv(args.csv, mode='a', index=False, header=False)
            else:
                df.to_csv(args.csv, mode='w', index=False, header=True)
            print(f"Result appended to {args.csv}")
        
        if args.visualize:
            try:
                metrics_chart, summary = save_metric_visualization(payload, args.image, args.viz_dir)
                print(f"Visualizations saved: {metrics_chart}, {summary}")
            except Exception as e:
                print(f"Warning: Failed to generate visualizations: {e}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
