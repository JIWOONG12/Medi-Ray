package com.example.MR.entity;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResultTb {
	private String xrayCode;
	private String ptCode;
	private double Atelectasis;
	private double Cardiomegaly; 
	private double Consolidation;
	private double Edema;
	private double Enlarged_Cardiomediastinum;
	private double Fracture;
	private double Lung_Lesion;
	private double Lung_Opacity;
	private double No_Finding;
	private double Pleural_Effusion;
	private double Pleural_Other;
	private double Pneumonia;
	private double Pneumothorax;
	private double Support_Devices;
}
