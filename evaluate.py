import pandas as pd


try:
    df = pd.read_csv('epochs.log')
except:
    print("epochs.log file not found in this folder")
    exit()

val_acc = round(max(df['val_acc']) * 100, 3)
val_auc = round(max(df['val_auc']), 5)
val_loss = round(min(df['val_loss']), 3)

print("Evaluation of the best model:")
print(f"Validation Loss:\t{val_loss}")
print(f"Validation Accuracy:\t{val_acc}%")
print(f"Validation AUC:\t\t{val_auc}")


print("\nEpoch\t\tAcc\t\tVal Acc\t\tLoss\t\tVal loss\tAUC\t\tVal AUC")
print("--------------------------------------------------------------------------------------------------------")

for i in range(len(df)):

    if df['val_auc'][i] == max(df['val_auc']):
        txt = f"[ {i + 1:02d} ]\t\t" \
            f"{round(df['acc'][i] * 100, 3)}*\t\t" \
            f"{round(df['val_acc'][i] * 100, 3)}*\t\t" \
            f"{round(df['loss'][i], 3)}*\t\t" \
            f"{round(df['val_loss'][i], 3)}*\t\t" \
            f"{round(df['auc'][i], 5)}*\t" \
            f"{round(df['val_auc'][i], 5)}*"

    else:
        txt = f"[ {i + 1:02d} ]\t\t" \
            f"{round(df['acc'][i] * 100, 3)}\t\t" \
            f"{round(df['val_acc'][i] * 100, 3)}\t\t" \
            f"{round(df['loss'][i], 3)}\t\t" \
            f"{round(df['val_loss'][i], 3)}\t\t" \
            f"{round(df['auc'][i], 5)}\t\t" \
            f"{round(df['val_auc'][i], 5)}"
    
    print(txt)
