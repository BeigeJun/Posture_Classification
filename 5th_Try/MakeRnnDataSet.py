import itertools
import pandas as pd

# 상태 리스트
states = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

# 3개의 인풋으로 구성된 모든 가능한 조합 생성
combinations = list(itertools.product(states.values(), repeat=3))

# 라벨 데이터를 저장할 리스트
data = []
last_label = None  # 마지막 라벨링 정보를 저장할 변수

print("Enter label for each combination (1: Safe, 2: Danger, 3: Caution, 4: Skip, 5: Edit last label):\n")

# 모든 조합을 순차적으로 출력하고 라벨 입력 받기
for idx, combo in enumerate(combinations):
    print(f'Combination {idx + 1}: {combo}')

    # FallDown이 2번 있는 경우
    if combo.count('FallDown') >= 2:
        label = 'Danger'
        print(f"Automatically labeling {combo} as 'Danger'.\n")
        data.append(list(combo) + [label])
        last_label = (combo, label)  # 마지막 라벨링 정보를 저장
        continue

    # Terrified가 2번 있는 경우
    if combo.count('Sleep') >= 2:
        label = 'Danger'
        print(f"Automatically labeling {combo} as 'Danger'.\n")
        data.append(list(combo) + [label])
        last_label = (combo, label)  # 마지막 라벨링 정보를 저장
        continue

    while True:
        label = input("Enter label (1: Safe, 2: Danger, 3: Caution, 4: Skip, 5: Edit last label): ")

        if label in ['1', '2', '3', '4', '5']:
            if label == '4':
                print("Skipping this combination.\n")
                break  # 해당 조합을 버리고 다음으로 넘어감

            elif label == '5':
                if last_label is not None:
                    # 최근에 입력한 조합과 라벨을 보여줌
                    print(f"Last labeled combination: {last_label[0]} with label {last_label[1]}")
                    new_label = input("Enter new label (1: Safe, 2: Danger, 3: Caution): ")

                    if new_label in ['1', '2', '3']:
                        # 라벨 수정
                        data[-1][3] = new_label  # 마지막 데이터의 라벨 수정
                        print("Last label updated.\n")
                    else:
                        print("Invalid input! Please enter 1, 2, or 3.")
                else:
                    print("No last label to edit.\n")
                continue

            # 입력한 라벨값 밀어주기
            if label == '1':
                label = 'Safe'  # 1을 2로 밀기
            elif label == '2':
                label = 'Danger'  # 2를 3으로 밀기
            elif label == '3':
                label = 'Caution'  # 3을 4로 밀기

            # 라벨을 추가
            data.append(list(combo) + [label])
            last_label = (combo, label)  # 마지막 라벨링 정보를 저장
            break
        else:
            print("Invalid input! Please enter 1, 2, 3, 4, or 5.")

# DataFrame으로 변환
df = pd.DataFrame(data, columns=['Input1', 'Input2', 'Input3', 'Label'])

# CSV 파일로 저장
df.to_csv('labeled_data.csv', index=False)
print("Data has been saved to 'labeled_data.csv'.")
