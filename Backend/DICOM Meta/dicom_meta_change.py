{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9b5f01f1-6b1e-4751-be2f-271770db2c96",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pydicom in c:\\users\\user\\anaconda3\\envs\\mr\\lib\\site-packages (3.0.1)\n",
      "Requirement already satisfied: SimpleITK in c:\\users\\user\\anaconda3\\envs\\mr\\lib\\site-packages (2.4.0)\n"
     ]
    }
   ],
   "source": [
    "# ! pip install simpleITK\n",
    "!pip install pydicom\n",
    "!pip install SimpleITK"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "78b95cfe-550b-44ba-9870-7d9b51c6ca6a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "환자 이름: Zendaya Maree \n",
      "환자 ID: 13150052\n",
      "환자 생년월일: 20180901\n",
      "환자 성별: F \n",
      "View Position: AP\n"
     ]
    }
   ],
   "source": [
    "import SimpleITK as sitk\n",
    "\n",
    "# DICOM 파일 경로\n",
    "dicom_file_path = \"C:/Users/USER/Downloads/13150052/7da88428-41454487-78f054a8-8fe1f558-7db1de82.dcm\"\n",
    "\n",
    "# DICOM 파일 읽기\n",
    "reader = sitk.ImageFileReader()\n",
    "reader.SetFileName(dicom_file_path)\n",
    "reader.LoadPrivateTagsOn()\n",
    "reader.ReadImageInformation()\n",
    "\n",
    "# 메타데이터 가져오기\n",
    "patient_name = reader.GetMetaData(\"0010|0010\")\n",
    "patient_id = reader.GetMetaData(\"0010|0020\")\n",
    "patient_birth_date = reader.GetMetaData(\"0010|0030\")\n",
    "patient_sex = reader.GetMetaData(\"0010|0040\")\n",
    "view_position = reader.GetMetaData(\"0018|5101\")\n",
    "\n",
    "# 메타데이터 출력\n",
    "print(f\"환자 이름: {patient_name}\")\n",
    "print(f\"환자 ID: {patient_id}\")\n",
    "print(f\"환자 생년월일: {patient_birth_date}\")\n",
    "print(f\"환자 성별: {patient_sex}\")\n",
    "print(f\"View Position: {view_position}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c8a48a3f-9eed-434a-ac0b-d262909cd331",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "파일 수정 완료\n"
     ]
    }
   ],
   "source": [
    "import pydicom\n",
    "from datetime import datetime\n",
    "\n",
    "def modify_patient_tags(dicom_path, patient_name=None, birth_date=None, sex=None, output_path=None):\n",
    "    \"\"\"\n",
    "    DICOM 파일의 환자 정보 태그를 수정하는 함수\n",
    "    \n",
    "    Parameters:\n",
    "    dicom_path (str): 원본 DICOM 파일 경로\n",
    "    patient_name (str): 환자 이름 (0010,0010)\n",
    "    birth_date (str): 생년월일 (0010,0030) - YYYYMMDD 형식\n",
    "    sex (str): 성별 (0010,0040) - M/F/O\n",
    "    output_path (str): 저장할 파일 경로 (없으면 원본 파일 덮어쓰기)\n",
    "    \"\"\"\n",
    "    # DICOM 파일 읽기\n",
    "    ds = pydicom.dcmread(dicom_path)\n",
    "    \n",
    "    # 환자 이름 수정 (0010,0010)\n",
    "    if patient_name is not None:\n",
    "        ds[0x0010, 0x0010].value = patient_name\n",
    "    \n",
    "    # 생년월일 수정 (0010,0030)\n",
    "    if birth_date is not None:\n",
    "        # 입력값이 올바른 형식인지 확인\n",
    "        try:\n",
    "            datetime.strptime(birth_date, '%Y%m%d')\n",
    "            ds[0x0010, 0x0030].value = birth_date\n",
    "        except ValueError:\n",
    "            raise ValueError(\"생년월일은 YYYYMMDD 형식이어야 합니다 (예: 19800101)\")\n",
    "    \n",
    "    # 성별 수정 (0010,0040)\n",
    "    if sex is not None:\n",
    "        if sex.upper() not in ['M', 'F', 'O']:\n",
    "            raise ValueError(\"성별은 'M', 'F', 또는 'O' 중 하나여야 합니다\")\n",
    "        ds[0x0010, 0x0040].value = sex.upper()\n",
    "    \n",
    "    # 수정된 파일 저장\n",
    "    if output_path:\n",
    "        ds.save_as(output_path)\n",
    "    else:\n",
    "        ds.save_as(dicom_path)\n",
    "    \n",
    "    return True\n",
    "\n",
    "# 사용 예시\n",
    "if __name__ == \"__main__\":\n",
    "    try:\n",
    "        # 단일 파일 수정 예시\n",
    "        modify_patient_tags(\n",
    "            dicom_path=\"C:/Users/USER/Downloads/13150052/df80fbd6-2c98232f-058c0bc0-42d427c8-7ff34ff5.dcm\",\n",
    "            patient_name=\"Zendaya Maree\",\n",
    "            birth_date=\"20180901\",\n",
    "            sex=\"F\",\n",
    "            output_path=\"C:/Users/USER/Downloads/13150052/df80fbd6-2c98232f-058c0bc0-42d427c8-7ff34ff5.dcm\"\n",
    "        )\n",
    "        print(\"파일 수정 완료\")\n",
    "                \n",
    "    except Exception as e:\n",
    "        print(f\"오류 발생: {str(e)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36812a4b-f244-4fee-895b-e65d56718294",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
