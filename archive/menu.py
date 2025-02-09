import os

print("Hello! What would you like to do?")

def menu():
    menu_entries = [
        "1. Sync with Raspberry Pi",
        "2. Run MNIST demonstration",
        "3. Run save-load demonstration"
    ]
    for entry in menu_entries:
        print(entry)

    choice = input("Enter your selection:")

    print(choice)
    if choice.isdigit(): # oddly enough works as it should
        match int(choice):
            case 1:
                print("Syncing...")
                os.system("rsync -av -e ssh ./* evdra@raspberrypi.local:~/soft")
            case 2:
                print("Running MNIST demonstration...")
                # HORRIBLE PRACTICE but this is just demonstration
                os.system("python3 ./tensor/classify.py")
            case 3:
                print("Running save/load demonstration")
                # Shits itself if ran through os.system but who cares rn, save-load works
                os.system("python3 ./tensor/save-and-load.py")
            case _:
                print ("Not a valid selection!")
                menu()
    else:
        print("Please input a valid integer!")
        menu()

menu()