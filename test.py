



def main():
    list_num = [1,2,3,4,5,6]

    xx = iter(list_num)
    for i in range(7):
        x = next(xx)

        print(x)

if __name__ == "__main__":
    main()

