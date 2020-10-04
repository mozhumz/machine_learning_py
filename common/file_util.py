import os
import common.common_util as common_util
def read_file(file_path:str):
    with open(file_path,mode='r',encoding='utf-8') as f:
        lines=f.readlines()
        f.close()
        return lines

def write_file(read_lines:[],out_file_path:str):
    read_lines=[line+'\n' for line in read_lines]
    with open(out_file_path,mode='w',encoding='utf-8') as f:
        f.writelines(read_lines)

def list_file(path:str,only_file=False):
    '''
    获取文件路径下的所有文件
    :param path:
    :param only_file: 是否只是返回文件，不返回文件夹
    :return:
    '''
    f_list=[]
    if only_file:
        for file_name in os.listdir(path=path):
            if os.path.isdir(path+file_name):
                continue
            f_list.append(file_name)
        return f_list
    return os.listdir(path=path)

def merge_text_files(in_path:str,out_path:str):
    with open(out_path,mode='w',encoding='utf-8') as f:
        for in_file in list_file(in_path):
            # 遍历每个文件 读取某个文件的所有行
            lines=read_file(in_path+in_file)
            read_lines=[]
            for line in lines:
                print("line:",line)
                words=line.split(" ")
                tokens=[]
                for word in words:
                    if common_util.check_str(word)==False:
                        print("wordErr:",word)
                        continue
                    word=common_util.trim_with_space_flag(word)
                    tokens.append(word)
                print('tokens:',tokens)
                read_lines.append(' '.join(tokens))
            read_lines=[line+'\n' for line in read_lines]
            f.writelines(read_lines)
        f.close()
    print('merge_text_files done!')
if __name__ == '__main__':
    path='G:\\bigdata\\badou\\00-data\\'
    merge_text_files(path+'data\\',path+'word2vec\\test\\news_merge.txt')
