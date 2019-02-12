
#
#
# # # Example/ testing code / debugging code
# logfile = os.path.join('articles_batch2.0','model0e0c27','model_logs','stdout.log')
# log_list = logfile2summary(logfile)
#
# import os
# import re
# def extract_floats_from_string(s):
#     ''' supporting function '''
#     l = []
#     for t in s.split():
#         try:
#             l.append(float(t))
#         except ValueError:
#             pass
#     return l
#
# # # Print all lines (debug only)
# # lines = []
# # with open(logfile) as search:
# #     # len(search)
# #     for line in search:
# #         line = line.rstrip()  # remove '\n' at end of line
# #         lines.append(line)
# #         if verbose: print(line)
#
# import string
# exclude_set = set(string.punctuation)
# exclude_set = ['"',':',',']
#
# log_summary = {}
#
# with open(logfile) as search:
#     # len(search)
#     for line in search:
#         # Strip all punctuation
#         line = line.rstrip()  # remove '\n' at end of line
#         line = ''.join(ch for ch in line if ch not in exclude_set)   # All other punctuation
#
#         searchterm = 'allennlp.common.util - Metrics'
#         if (searchterm + ' ') in line:
#             print(line)
#             line2 = line
#
# import string
#
#

# # Imports!
def logfile2summary(logfile,order=None,verbose=False):
    '''Converts logfile output from allennlp into a list. Can then save to summary csv'''

    # Default settings
    if not order: order = ['datetime','best_validation_loss','training_loss','best_validation_accuracy','training_accuracy']
    # impotrs

    import os
    import re

    # functions
    def extract_floats_from_string(s):
        ''' supporting function '''
        l = []
        for t in s.split():
            try:
                l.append(float(t))
            except ValueError:
                pass
        return l

    # # Print all lines (debug only)
    # lines = []
    # with open(logfile) as search:
    #     # len(search)
    #     for line in search:
    #         line = line.rstrip()  # remove '\n' at end of line
    #         lines.append(line)
    #         if verbose: print(line)

    import string
    exclude_set = set(string.punctuation)
    exclude_set = ['"',':',',']

    log_summary = {}

    with open(logfile) as search:
        # len(search)
        for line in search:
            # Strip all punctuation
            line = line.rstrip()  # remove '\n' at end of line
            line = ''.join(ch for ch in line if ch not in exclude_set)   # All other punctuation

            searchterm = 'best_validation_loss'
            if (searchterm + ' ') in line:
                if verbose: print(line)
                log_summary[searchterm] = (extract_floats_from_string(line))

            searchterm = 'best_validation_accuracy'
            if (searchterm + ' ') in line:
                if verbose: print(line)
                log_summary[searchterm] = (extract_floats_from_string(line))

            searchterm = 'training_loss'
            if (searchterm + ' ') in line:
                if verbose: print(line)
                log_summary[searchterm] = (extract_floats_from_string(line))

            searchterm = 'training_accuracy'
            if (searchterm + ' ') in line:
                if verbose: print(line)
                log_summary[searchterm] = (extract_floats_from_string(line))

            # Date and time of simulation
            searchterm = 'allennlp.common.util - Metrics'
            if (searchterm + ' ') in line:
                if verbose: print(line)
                line2 = line
                line3 = ''.join(ch for ch in line2 if ch not in set(string.punctuation))   # All other punctuation
                log_summary['datetime'] = line3[0:18]

            # searchterm = 'training_duration'
            # if (searchterm + ' ') in line:
            #     if verbose: print(line)
            #     log_summary[searchterm] = (extract_floats_from_string(line))[0]

    # Keep only first entry in dictionary
    for k in log_summary.keys():
        if not k == 'datetime':
            log_summary[k] = (log_summary[k])[0]

    # print(log_summary)
    log_list = [log_summary['datetime'],log_summary['best_validation_loss'],log_summary['training_loss'],log_summary['best_validation_accuracy'],log_summary['training_accuracy']]
    return log_list




import glob
import os

myorder = ['datetime','best_validation_loss','training_loss','best_validation_accuracy','training_accuracy']

csv_rows = []
subfolders = sorted(glob.glob('articles*'))
for sf in subfolders:
    subfolders2 = sorted(glob.glob(os.path.join(sf,'model*')))
    for sf2 in subfolders2:
        print (sf2)
        folder_struct = sf2.split('/')
        logfile = os.path.join(sf2,'model_logs','stdout.log')
        if os.path.isfile(logfile):
            logfile_list = logfile2summary(logfile,myorder)
            csv_row = [folder_struct[0],folder_struct[1],''] + logfile_list
            csv_rows.append(csv_row)
        else:
            print('Warning, log file {} not found'.format(logfile))
            csv_row = [folder_struct[0],folder_struct[1],''] + ['error']
            csv_rows.append(csv_row)


    # Now do old entropy folders
    subfolders2b = sorted(glob.glob(os.path.join(sf,'*entropy*')))
    for sf2b in subfolders2b:
        subfolders2 = sorted(glob.glob(os.path.join(sf2b,'model*')))
        for sf2 in subfolders2:
            print (sf2)
            folder_struct = sf2.split('/')
            logfile = os.path.join(sf2,'model_logs','stdout.log')
            if os.path.isfile(logfile):
                logfile_list = logfile2summary(logfile,myorder)

                csv_row = [folder_struct[0],folder_struct[2],folder_struct[1]] + logfile_list
                csv_rows.append(csv_row)
            else:
                print('Warning, log file {} not found'.format(logfile))
                csv_row = [folder_struct[0],folder_struct[2],folder_struct[1]] + ['error']
                csv_rows.append(csv_row)



# Write CSV
import csv
with open('model_performance.csv', mode='w') as file:
    mywriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # mywriter.writerow('batch','model','entropy_settings','datetime','best_validation_loss','training_loss','best_validation_accuracy','training_accuracy')
    mywriter.writerow(['batch','model','entropy_settings'] + myorder)
    for i,row in enumerate(csv_rows):
        mywriter.writerow(csv_rows[i])
