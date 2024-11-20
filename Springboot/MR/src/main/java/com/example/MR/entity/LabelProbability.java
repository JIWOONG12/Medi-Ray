package com.example.MR.entity;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class LabelProbability {
    private String labelName;      // 라벨 이름 (예: 'Atelectasis', 'Consolidation' 등)
    private Double probability;    // 해당 라벨의 확률 값 (예: 87.5)

    // 생성자: LabelProbability 객체를 만들 때 라벨 이름과 확률 값을 설정
    public LabelProbability(String labelName, Double probability) {
        this.labelName = labelName;
        this.probability = probability;
    }
}
