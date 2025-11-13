#!/usr/bin/env python3
"""
Script to migrate TensorFlow 1.x code to TensorFlow 2.x
Updates Keras imports and API calls
"""
import os
import re
from pathlib import Path


def update_file_content(content):
    """Update file content with TF2 and modern API changes"""

    # Update Keras imports
    content = re.sub(r'from keras\.', r'from tensorflow.keras.', content)
    content = re.sub(r'import keras\b', r'import tensorflow.keras as keras', content)

    # Update optimizer lr parameter to learning_rate
    content = re.sub(r'Adam\(lr=', r'Adam(learning_rate=', content)
    content = re.sub(r'RMSprop\(lr=', r'RMSprop(learning_rate=', content)
    content = re.sub(r'SGD\(lr=', r'SGD(learning_rate=', content)

    # Add verbose=0 to predict calls if not already present
    # This handles the new TF2 warning about verbosity
    content = re.sub(
        r'\.predict\(([^)]+)\)(?!\s*,\s*verbose)',
        r'.predict(\1, verbose=0)',
        content
    )

    # Fix gym reset() API - newer gym returns (obs, info) tuple
    # This is a simple pattern - might need manual review
    content = re.sub(
        r'state = env\.reset\(\)',
        r'state = env.reset()\n        if isinstance(state, tuple):\n            state = state[0]',
        content
    )
    content = re.sub(
        r'observe = env\.reset\(\)',
        r'observe = env.reset()\n        if isinstance(observe, tuple):\n            observe = observe[0]',
        content
    )

    # Fix gym step() API - newer gym returns 5-tuple (obs, reward, terminated, truncated, info)
    # Old: next_state, reward, done, info = env.step(action)
    # This needs manual review as the logic might need adjustment

    return content


def process_file(filepath):
    """Process a single Python file"""
    print(f"Processing {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        updated_content = update_file_content(content)

        if updated_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"  ✓ Updated {filepath}")
            return True
        else:
            print(f"  - No changes needed for {filepath}")
            return False
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False


def main():
    """Main migration function"""
    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))

    # Exclude this migration script
    python_files = [f for f in python_files if f.name != "migrate_to_tf2.py"]

    print(f"Found {len(python_files)} Python files to process\n")

    updated_count = 0
    for filepath in python_files:
        if process_file(filepath):
            updated_count += 1

    print(f"\n{'='*60}")
    print(f"Migration complete!")
    print(f"Updated {updated_count} out of {len(python_files)} files")
    print(f"{'='*60}")
    print("\nNOTE: Some files may need manual review:")
    print("  - TensorFlow 1.x Session API usage")
    print("  - Custom training loops")
    print("  - Gym step() API (done/terminated/truncated)")


if __name__ == "__main__":
    main()
