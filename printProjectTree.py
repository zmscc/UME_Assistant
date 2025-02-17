import os

def print_directory_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

if __name__ == "__main__":
    # project_path = os.getcwd()  # 获取当前工作目录，即项目根目录
    project_path = r'C:\deepLearningCode\UME_Chat\libs\chatchat-server\chatchat'
    print_directory_structure(project_path)