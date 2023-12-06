from venv import create


def create_y(df):
    var_names = []
    for i in range(26, 38):
        name = df.iloc[:,i].name
        vals = df.iloc[:,i]

        var_names.append(name)

        globals()[name] = vals

    for i in range(2, 14):
        name = df.iloc[:,i].name
        vals = df.iloc[:,i]

        var_names.append(name)

        globals()[name] = vals


    global ADD 
    global DIV
    global SUB
    global MUL

    global SUM
    
    ADD = ADD1 + ADD2 + ADD3
    DIV = DIV1 + DIV2 + DIV3
    SUB = SUB1 + SUB2 + SUB3
    MUL = MUL1 + MUL2 + MUL3
    
    SUM = ADD + DIV + SUB + MUL

    var_names += ['ADD', 'DIV', "MUL", 'SUB', "SUM"]
    
    return var_names




def make_X(df, y_name, use_math_operations=True):
    
    if use_math_operations:
        return new_math_operations_X(df, y_name)
       # return math_operations_X(df, y_name)
    else:
        return math_operations_X(df, 'ADD1')
    

def math_operations_X(df, y_name):


    if 'O_' in y_name:
        y_name = y_name.replace('O_', '')

    if y_name != 'SUM':

        diff_group = create_diff_group()
    
        number_level = None

        difficulties = ['ADD', 'SUB', 'MUL', 'DIV']

        columns_to_drop = ['n_sum']



        try: 
            number_level = int(y_name[-1])
        except ValueError:
            y_name = y_name + '1'
            number_level = 1

        if 'ACC' in y_name:
            base_name = y_name[4:-1]
        else:
            base_name = y_name[:-1]

        columns_to_drop.append(y_name)    
            
        




        more_difficult = difficulties[difficulties.index(base_name):]
        #print(more_difficult)

        for level in more_difficult:
            columns_to_drop.extend([level+'1', level+'2', level+'3'])
            columns_to_drop.extend(['ACC_'+level+'1', 'ACC_'+level+'2', 'ACC_'+level+'3'])
            columns_to_drop.extend(['RT_'+level+'1', 'RT_'+level+'2', 'RT_'+level+'3'])

            columns_to_drop.extend(['RT_'+level+'1', 'RT_'+level+'2', 'RT_'+level+'3'])

            columns_to_drop.extend(diff_group[level])
            columns_to_drop.extend(['O_' + level])
            #print('O_' + level)

            #columns_to_drop.append('ACC_'+level)
           # columns_to_drop.append('N_'+level)



        if number_level is not None:
            if number_level == 1:
                columns_to_drop.extend(['ACC_' + base_name+'2','ACC_' +  base_name+'3'])
                columns_to_drop.extend(['RT_' + base_name+'2','RT_' +  base_name+'3'])
                columns_to_drop.extend([base_name+'2', base_name+'3'])
                columns_to_drop.extend(diff_group[base_name])
                
                if 'ACC' not in y_name:
                    columns_to_drop.append('ACC_' + base_name + '1')
                    columns_to_drop.append('RT_' + base_name + '1')
                    
                else:
                    columns_to_drop.append(base_name + '1')
                
            elif number_level == 2:
                columns_to_drop.extend(['ACC_' + base_name+'3'])
                columns_to_drop.extend(['RT_' + base_name+'3'])
                columns_to_drop.extend([base_name+'3'])
                columns_to_drop.extend(diff_group[base_name])


                
                if 'ACC' not in y_name:
                    columns_to_drop.append('ACC_' + base_name + '2')
                    columns_to_drop.append('RT_' + base_name + '2')
                    
                else:
                    columns_to_drop.append(base_name + '2')
                
            elif number_level == 3:

                columns_to_drop.extend(['DIFF_'+base_name+'31', 'DIFF_'+base_name+'32'])
                columns_to_drop.extend(['DIFF_ACC_'+base_name+'31', 'DIFF_ACC_'+base_name+'32'])
                columns_to_drop.extend(['DIFF_RT_'+base_name+'31', 'DIFF_RT_'+base_name+'32'])


                if 'ACC' not in y_name:
                    columns_to_drop.append('ACC_' + base_name + '3')
                    columns_to_drop.append('RT_' + base_name + '3')
                    
                else:
                    columns_to_drop.append(base_name + '3')

        
    
    else:
        columns_to_drop = ['n_sum']
        


        
   # print(columns_to_drop)
    return df.drop(columns=columns_to_drop)
    


def create_diff_group():
  return {'ADD': ['DIFF_ACC_ADD21',
 'DIFF_ACC_ADD32',
 'DIFF_ACC_ADD31',
 'DIFF_RT_ADD21',
 'DIFF_RT_ADD32',
 'DIFF_RT_ADD31',
 'DIFF_ADD21',
 'DIFF_ADD32',
 'DIFF_ADD31'],
    'DIV': ['DIFF_ACC_DIV21',
 'DIFF_ACC_DIV32',
 'DIFF_ACC_DIV31',
 'DIFF_RT_DIV21',
 'DIFF_RT_DIV32',
 'DIFF_RT_DIV31',
 'DIFF_DIV21',
 'DIFF_DIV32',
 'DIFF_DIV31'],
    'MUL': ['DIFF_ACC_MUL21',
 'DIFF_ACC_MUL32',
 'DIFF_ACC_MUL31',
 'DIFF_RT_MUL21',
 'DIFF_RT_MUL32',
 'DIFF_RT_MUL31',
 'DIFF_MUL21',
 'DIFF_MUL32',
 'DIFF_MUL31'],
    'SUB': ['DIFF_ACC_SUB21',
 'DIFF_ACC_SUB32',
 'DIFF_ACC_SUB31',
 'DIFF_RT_SUB21',
 'DIFF_RT_SUB32',
 'DIFF_RT_SUB31',
 'DIFF_SUB21',
 'DIFF_SUB32',
 'DIFF_SUB31']}

def new_math_operations_X(df, y_name):

    if y_name == 'O_23':
        return df.drop(columns=['O_23'])

    if y_name == 'O_12':
        columns_to_drop = ['ACC_ADD3', 'ACC_DIV3', 'ACC_MUL3', 'ACC_SUB3',
                            'RT_ADD3', 'RT_DIV3', 'RT_MUL3', 'RT_SUB3',
                            'ADD3', 'DIV3', 'MUL3', 'SUB3',
                            'DIFF_ACC_ADD32', 'DIFF_ACC_ADD31', 'DIFF_RT_ADD32',
 'DIFF_RT_ADD31', 'DIFF_ADD32', 'DIFF_ADD31', 'DIFF_ACC_DIV32', 'DIFF_ACC_DIV31', 'DIFF_RT_DIV32', 'DIFF_RT_DIV31', 'DIFF_DIV32',
 'DIFF_DIV31','DIFF_ACC_MUL32', 'DIFF_ACC_MUL31',  'DIFF_RT_DIV32', 'DIFF_RT_DIV31','DIFF_DIV32', 'DIFF_DIV31', 'DIFF_ACC_SUB32',
 'DIFF_ACC_SUB31','DIFF_RT_SUB32', 'DIFF_RT_SUB31','DIFF_SUB32', 'DIFF_SUB31', 'O_12', 'O_23', 'DIFF_MUL32', 'DIFF_MUL31',
  'DIFF_RT_MUL32', 'DIFF_RT_MUL31']
        return df.drop(columns = columns_to_drop)

        



    if y_name != 'SUM':

        diff_group = create_diff_group()
    
        number_level = None

        difficulties = ['ADD', 'SUB', 'MUL', 'DIV']

        columns_to_drop = ['n_sum']

        if 'O_' in y_name:
            columns_to_drop.append(y_name)
            y_name = y_name.replace('O_', '')
            



        try: 
            number_level = int(y_name[-1])
        except ValueError:
            y_name = y_name + '1'
            number_level = 1

        if 'ACC' in y_name:
            base_name = y_name[4:-1]
        else:
            base_name = y_name[:-1]

        columns_to_drop.append(y_name)    
            



        if number_level is not None:
            if number_level == 1:
                columns_to_drop.extend(['ACC_' + base_name+'2','ACC_' +  base_name+'3'])
                columns_to_drop.extend(['RT_' + base_name+'2','RT_' +  base_name+'3'])
                columns_to_drop.extend([base_name+'2', base_name+'3'])
                columns_to_drop.extend(diff_group[base_name])
                
                if 'ACC' not in y_name:
                    columns_to_drop.append('ACC_' + base_name + '1')
                    columns_to_drop.append('RT_' + base_name + '1')
                    
                else:
                    columns_to_drop.append(base_name + '1')
                
            elif number_level == 2:
                columns_to_drop.extend(['ACC_' + base_name+'3'])
                columns_to_drop.extend(['RT_' + base_name+'3'])
                columns_to_drop.extend([base_name+'3'])
                columns_to_drop.extend(diff_group[base_name])


                
                if 'ACC' not in y_name:
                    columns_to_drop.append('ACC_' + base_name + '2')
                    columns_to_drop.append('RT_' + base_name + '2')
                    
                else:
                    columns_to_drop.append(base_name + '2')
                
            elif number_level == 3:

                columns_to_drop.extend(['DIFF_'+base_name+'31', 'DIFF_'+base_name+'32'])
                columns_to_drop.extend(['DIFF_ACC_'+base_name+'31', 'DIFF_ACC_'+base_name+'32'])
                columns_to_drop.extend(['DIFF_RT_'+base_name+'31', 'DIFF_RT_'+base_name+'32'])


                if 'ACC' not in y_name:
                    columns_to_drop.append('ACC_' + base_name + '3')
                    columns_to_drop.append('RT_' + base_name + '3')
                    
                else:
                    columns_to_drop.append(base_name + '3')

        
    
    else:
        columns_to_drop = ['n_sum']
        


        
   # print(columns_to_drop)
    return df.drop(columns=columns_to_drop)