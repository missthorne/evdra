print("hello, this is a file I shall use to feck about in python until I figure out what's what.")

print("what is it that you'd like to do?")

def menu():
    menu_entries = [
        "1. See documentation",
        "2. idk yet just checking if this works",
        "3. help?"
    ]
    for entry in menu_entries:
        print(entry)

    choice = input("Enter your selection:")

    print(choice)
    if choice.isdigit(): # oddly enough works as it should
        match int(choice):
            case 1:
                print("show docs here (maybe in separate file?")
            case 2:
                print("idk but I think 2 works")
            case 3:
                print("no help")
            case _:
                print ("Not a valid selection!")
                menu()
    else:
        print("Please input a valid integer!")
        menu()

menu()