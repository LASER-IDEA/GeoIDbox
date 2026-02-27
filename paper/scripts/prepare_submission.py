#!/usr/bin/env python3
"""
Prepare IEEE TIM paper submission.
Checks completeness and generates submission package.
"""

import os
import shutil
from pathlib import Path

# Required files for submission
REQUIRED_FILES = {
    'source': [
        'main.tex',
        'method.tex',
        'experiment.tex',
        'references.bib',
    ],
    'figures': [
        'fig1_method_comparison.pdf',
        'fig2_loso_results.pdf',
        'fig3_ablation.pdf',
        'fig4_architecture.pdf',
        'fig5_curriculum.pdf',
        'fig6_spatial_map.pdf',
    ],
    'tables': [
        'table_main_results.tex',
        'table_loso.tex',
        'table_ablation.tex',
        'table_architecture.tex',
        'table_dataset.tex',
    ]
}

def check_file_exists(filepath):
    """Check if a file exists."""
    return os.path.exists(filepath)

def check_completeness():
    """Check if all required files are present."""
    print("=" * 60)
    print("IEEE TIM Paper Submission Checklist")
    print("=" * 60)
    
    all_good = True
    
    # Check source files
    print("\n📄 Source Files:")
    for f in REQUIRED_FILES['source']:
        path = f"./{f}"
        exists = check_file_exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        if not exists:
            all_good = False
    
    # Check figures
    print("\n🖼️  Figures (PDF for vector quality):")
    for f in REQUIRED_FILES['figures']:
        path = f"./figures/{f}"
        exists = check_file_exists(path)
        status = "✅" if exists else "❌"
        size = os.path.getsize(path) / 1024 if exists else 0
        print(f"  {status} {f} ({size:.1f} KB)")
        if not exists:
            all_good = False
    
    # Check tables
    print("\n📊 Tables:")
    for f in REQUIRED_FILES['tables']:
        path = f"./tables/{f}"
        exists = check_file_exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        if not exists:
            all_good = False
    
    return all_good

def generate_submission_package():
    """Generate submission package."""
    print("\n" + "=" * 60)
    print("Generating Submission Package")
    print("=" * 60)
    
    # Create submission directory
    submission_dir = "submission_package"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    # Copy source files
    os.makedirs(f"{submission_dir}/source")
    for f in REQUIRED_FILES['source']:
        shutil.copy(f"./{f}", f"{submission_dir}/source/")
    
    # Copy figures
    os.makedirs(f"{submission_dir}/figures")
    for f in REQUIRED_FILES['figures']:
        shutil.copy(f"./figures/{f}", f"{submission_dir}/figures/")
    
    # Copy tables
    os.makedirs(f"{submission_dir}/tables")
    for f in REQUIRED_FILES['tables']:
        shutil.copy(f"./tables/{f}", f"{submission_dir}/tables/")
    
    print(f"✅ Submission package created: {submission_dir}/")
    print(f"   - Source files: {len(REQUIRED_FILES['source'])}")
    print(f"   - Figures: {len(REQUIRED_FILES['figures'])}")
    print(f"   - Tables: {len(REQUIRED_FILES['tables'])}")

def print_paper_stats():
    """Print paper statistics."""
    print("\n" + "=" * 60)
    print("Paper Statistics")
    print("=" * 60)
    
    # Count lines in tex files
    total_lines = 0
    for f in REQUIRED_FILES['source']:
        if f.endswith('.tex'):
            with open(f"./{f}", 'r') as file:
                lines = len(file.readlines())
                total_lines += lines
                print(f"  {f}: {lines} lines")
    
    print(f"\n  Total LaTeX lines: {total_lines}")
    
    # Figure sizes
    print("\n  Figure sizes (PDF):")
    total_size = 0
    for f in REQUIRED_FILES['figures']:
        path = f"./figures/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            total_size += size
            print(f"    {f}: {size:.1f} KB")
    print(f"  Total figure size: {total_size:.1f} KB")

def main():
    """Main function."""
    # Check completeness
    all_good = check_completeness()
    
    if all_good:
        print("\n✅ All required files are present!")
        
        # Print statistics
        print_paper_stats()
        
        # Generate submission package
        generate_submission_package()
        
        print("\n" + "=" * 60)
        print("Submission Checklist Summary")
        print("=" * 60)
        print("""
Before submitting to IEEE TIM, verify:

□ Manuscript formatted according to IEEE TIM guidelines
□ All figures in vector format (PDF) or high-res (300+ DPI)
□ Tables follow IEEE format
□ References properly formatted
□ Abstract within 200 words
□ Keywords included
□ Author affiliations correct
□ No identifying info for double-blind review
□ Supplementary material prepared (if applicable)

To compile:
  cd submission_package/source
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

Or use: latexmk -pdf main.tex
        """)
    else:
        print("\n❌ Some required files are missing!")
        print("Run the following to generate missing files:")
        print("  python generate_figures_simple.py")
        print("  python generate_tables.py")

if __name__ == "__main__":
    main()
