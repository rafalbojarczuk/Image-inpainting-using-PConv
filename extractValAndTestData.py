import os
import shutil

VAL_PERCENT = 10
TEST_PERCENT = 5
DRY_RUN = False

working_directory = os.getcwd()
train_dir = os.path.join(working_directory, "data_256")
val_dir = os.path.join(working_directory, "val_256")
test_dir = os.path.join(working_directory, "test_256")

if not os.path.isdir(train_dir):
    print("train_dir not found, aborting")
    exit()

if not os.path.isdir(val_dir):
    print("Creating val_dir")
    os.mkdir(val_dir)
elif len(os.listdir(val_dir)):
    print("val_dir not empty, aborting")
    exit()

if not os.path.isdir(test_dir):
    print("Creating test_dir")
    os.mkdir(test_dir)
elif len(os.listdir(test_dir)):
    print("test_dir not empty, aborting")
    exit()

print('{0: <20}'.format("category"),
      '{0: <5}'.format("train"),
      '{0: <5}'.format("val"),
      '{0: <5}'.format("test"))
for letter in os.listdir(train_dir):
    letterDir = os.path.join(train_dir, letter)
    for category in os.listdir(letterDir):
        categoryDir = os.path.join(letterDir, category)
        files = os.listdir(categoryDir)
        noValFiles = int(VAL_PERCENT * len(files) / 100)
        noTestFiles = int(TEST_PERCENT * len(files) / 100)
        print('{0: <20}'.format(category),
              '{0: <5}'.format(len(files) - noValFiles - noTestFiles),
              '{0: <5}'.format(noValFiles),
              '{0: <5}'.format(noTestFiles))
        valMoved = 0
        testMoved = 0
        for file in files:
            filePath = os.path.join(categoryDir, file)
            if valMoved < noValFiles:
                valMoved += 1
                if not DRY_RUN:
                    os.rename(filePath, os.path.join(val_dir, category+file))
            elif testMoved < noTestFiles:
                testMoved += 1
                if not DRY_RUN:
                    os.rename(filePath, os.path.join(test_dir, category+file))
            else:
                break
# for root, dirs, files in os.walk(train_dir):
#     folder_name = root.split("\\")[-1]
#     if folder_name == "data_256":
#         continue
#     for name in dirs:
#         files = [f for f in os.listdir(name)]
#         print(name)
