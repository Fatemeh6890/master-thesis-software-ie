labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application_Creation', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application_Mention', 'O', 'O']
group_labels = ['Application',
 'SoftwareCoreference',
 'OperatingSystem',
 'ProgrammingEnvironment',
 'PlugIn']
 
group_label_list = []
for group_label in group_labels:
    for label in labels:
        if label == 'O':
            group_label_list.append('O')
        else:
            if group_label in label:
                group_label_list.append(group_label)
            
                
print(dataset['labels'][0])
print(group_label_list)