
def clearv():
    with open('C:/Users/lenovo/Desktop/Test province/v.txt', "r+") as f:
        read_data = f.read()
        f.seek(0)
        f.truncate()  
    with open('C:/Users/lenovo/Desktop/Test province/vp.txt', "r+") as f:
        read_data = f.read()
        f.seek(0)
        f.truncate()  
    with open('C:/Users/lenovo/Desktop/Test province/score.txt', "r+") as f:
        read_data = f.read()
        f.seek(0)
        f.truncate()
    with open('C:/Users/lenovo/Desktop/Test province/score-f.txt', "r+") as f:
        read_data = f.read()
        f.seek(0)
        f.truncate()

# clearv()
