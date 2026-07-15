import pandas as pd
for f in ['Test 1 orari e percentuali.xlsx', 'Test 2 orari e percentuali.xlsx']:
    print('===', f, '===')
    df = pd.read_excel(r'.\Flowsense V2\\' + f)
    print('COLUMNS:', list(df.columns))
    print(df.to_string())
    print()
