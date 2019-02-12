
# # Imports!
def logfile2summary(logfile,verbose=False):
    '''Converts logfile output from allennlp into a list. Can then save to summary csv'''
    # Returned order is:
    import os
    import re
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

            # searchterm = 'training_duration'
            # if (searchterm + ' ') in line:
            #     if verbose: print(line)
            #     log_summary[searchterm] = (extract_floats_from_string(line))[0]

    # Keep only first entry in dictionary
    for k in log_summary.keys():
        log_summary[k] = (log_summary[k])[0]

    log_list = [log_summary['best_validation_loss'],log_summary['training_loss'],log_summary['best_validation_accuracy'],log_summary['training_accuracy']]
    return log_list


# logfile = os.path.join('articles_batch2.0','model0e0c27','model_logs','stdout.log')
# log_list = logfile2summary(logfile)

import glob
import os

csv_rows = []
subfolders = glob.glob('articles*')
for sf in subfolders:
    subfolders2 = glob.glob(os.path.join(sf,'model*'))
    for sf2 in subfolders2:
        print (sf2)
        logfile = os.path.join(sf2,'model_logs','stdout.log')
        if os.path.isfile(logfile):
            logfile_list = logfile2summary(logfile)
            folder_struct = sf2.split('/')
            csv_row = [folder_struct[0],folder_struct[1],'',logfile_list]
            csv_rows.append(csv_row)
        else:
            print('Error, log file {} not found'.format(logfile))
            csv_rows.append('error')


    # Now do old entropy folders
    subfolders2b = glob.glob(os.path.join(sf,'*entropy*'))
    for sf2b in subfolders2b:
        subfolders2 = glob.glob(os.path.join(sf2b,'model*'))
        for sf2 in subfolders2:
            print (sf2)

            logfile = os.path.join(sf2,'model_logs','stdout.log')
            if os.path.isfile(logfile):
                logfile_list = logfile2summary(logfile)
                folder_struct = sf2.split('/')
                csv_row = [folder_struct[0],folder_struct[2],folder_struct[1],logfile_list]
                csv_rows.append(csv_row)
            else:
                print('Error, log file {} not found'.format(logfile))
                csv_rows.append('error')


#
# # Convert to csv format
# import csv
# with open('employee_file.csv', mode='w') as employee_file:
#     employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#
#     employee_writer.writerow(['John Smith', 'Accounting', 'November'])
#     employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
