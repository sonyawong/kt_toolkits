import pandas as pd


def read_data_from_csv(read_file, write_file):
    df = pd.read_csv(read_file)
    ui_df = df.groupby(['user_id'], sort=False)

    user_inters = []
    for ui in ui_df:
        tmp_user, tmp_inter = ui[0], ui[1]
        tmp_seq_len = len(tmp_inter)
        tmp_problems = [str(x) for x in list(tmp_inter['log_id'])]
        tmp_skills = [str(x) for x in list(tmp_inter['sequence_id'])]
        tmp_ans = ['1' if x == 1.0 else '0' for x in list(tmp_inter['correct'])]
        tmp_start_time = list('NA' for i in range(tmp_seq_len))
        tmp_response_cost = list('NA' for i in range(tmp_seq_len))

        user_inters.append(
            [[str(tmp_user), str(tmp_seq_len)], tmp_problems, tmp_skills, tmp_ans, tmp_start_time, tmp_response_cost])

    write_txt(write_file, user_inters)

    return

def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            for d in dd:
                f.write(','.join(d) + '\n')

if __name__ == '__main__':
    read_data_from_csv('../data/assist2015/2015_100_skill_builders_main_problems.csv', '../data/assist2015/data.txt')