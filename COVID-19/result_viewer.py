def handle_user_input():
    menu_text = """
Enter an option by its number:
1. Rank models using bar plots.
2. Create cross-validation tables.
3. Interactive prediction viewer.
4. Check validation folds for a location.
5. Exit.
"""
    while True:
        try:
            option = int(input(menu_text))
        except ValueError:
            print("Option must be an integer.")
            continue

        if option == 5:
            raise SystemExit

        if option in range(1,5):
            return option
        else:
            print("Invalid option.")

def main():
    usr_chiose = handle_user_input()
    print(usr_chiose)

if __name__ == "__main__":
    main()
