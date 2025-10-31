#!/usr/bin/env python3
"""
Generate a secure secret key for the YOLO Web Demo
"""
import secrets
import os

def generate_secret_key(length=32):
    """Generate a cryptographically secure secret key"""
    return secrets.token_hex(length)

def main():
    """Generate and display a secret key"""
    print("YOLO Web Demo - Secret Key Generator")
    print("=" * 40)

    # Generate a secure secret key
    secret_key = generate_secret_key()

    print(f"Generated SECRET_KEY: {secret_key}")
    print()
    print("Add this to your .env file:")
    print(f"SECRET_KEY={secret_key}")
    print()
    print("IMPORTANT:")
    print("- Keep this key secret and secure")
    print("- Don't commit it to version control")
    print("- Use a different key for each environment")
    print("- Regenerate if key may be compromised")

if __name__ == "__main__":
    main()