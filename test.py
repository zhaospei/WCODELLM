from sklearn.metrics import accuracy_score

# true_labels = [test_dict['label'] for test_dict in df_test_dict
true_labels = []
for i in range(768):
    true_labels.append(0)

for i in range(872):
    true_labels.append(1)



predictions = []
for range in range(1640):
    predictions.append(0)
# for text in generated_texts:
#     if 'yes' in text or 'Yes' in text:
#         predictions.append(1)
#     else:
#         predictions.append(0)

print(sum(predictions))

accuracy = accuracy_score(true_labels, predictions)
from sklearn.metrics import classification_report
print(classification_report(true_labels, predictions))
