from src.lr0.data import prepare_split
if __name__ == "__main__":
    split = prepare_split(0)
    print("Разбиение создано:", {k: len(split[k]) for k in ("train", "val", "test")})
