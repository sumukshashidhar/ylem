import ylem

cleaner = ylem.Cleaner("nano")
text = "This is some sample text from the internet."
out = cleaner(text, max_new_tokens=32, do_sample=False)
print(out)