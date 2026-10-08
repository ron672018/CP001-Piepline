import fitz  # PyMuPDF

dummy_resumes = {
    "resume_alice_williams.pdf": (
        "Alice Williams\n"
        "Email: alice.w@example.com | Phone: 555-123-4567\n"
        "Location: Austin, TX\n\n"
        "Senior DevOps Engineer with 8 years of experience. Expert in Docker, Kubernetes, CI/CD pipelines, and AWS infrastructure."
    ),
    "resume_bob_smith.pdf": (
        "Bob Smith\n"
        "Email: bsmith_dev@test.org | Phone: 555-987-6543\n"
        "Location: Boston, MA\n\n"
        "Backend Software Developer specializing in Python, Django, and PostgreSQL. Passionate about building scalable REST APIs."
    )
}

if __name__ == "__main__":
    for filename, content in dummy_resumes.items():
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), content, fontsize=12)
        doc.save(filename)
        doc.close()
        print(f"Created sample resume: {filename}")