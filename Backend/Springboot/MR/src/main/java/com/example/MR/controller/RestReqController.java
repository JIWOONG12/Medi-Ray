package com.example.MR.controller;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.example.MR.mapper.MRMapper;

@RestController
public class RestReqController {
	
	@Autowired
    private MRMapper mapper;
	
	// Bean에 대한 의존성 주입
	@Autowired
	private WebClient WebClient;
	
	// post 방식으로 메세지, 이미지를 전송
	@PostMapping("/upload")
	public String dicomUpload(MultipartFile file, String message) {
		// post형식으로 전송하기 위해서 멀티파트 폼 데이터 형식의 바디 전송
		// 자바코드 안에는 입력양식을 만들기 어렵기 때문에
		MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder(); // 멀티파트 폼 데이터를 구성
		bodyBuilder.part("message", message); // 폼 데이터
		bodyBuilder.part("file", file.getResource()); // 폼 데이터, 파일
		// fastapi 주소로 요청을 할 때 멀티폼데이터로 컨텐츠가 들어가고
		// BodyBuilder에 담은 메세지와 file을 BodyInserters를 이용하여 Body에 넣는다.
		// retrieve를 통해서 결과를 받는다.
		// 받은 타입은 body2-mono라는 클래스를 이용해서 String 타입으로 변환
		String result = WebClient.post().uri("/dicom/dupload")// POST 방식으로 요청. 엔드포인트는 /dupload
				.contentType(MediaType.MULTIPART_FORM_DATA) // 파일이 전송
				.body(BodyInserters.fromMultipartData(bodyBuilder.build())) // 폼데이터를 요청 본문으로 설정
				.retrieve()// 요청을 실행하고 응답을 받음
				.bodyToMono(String.class) // 본문을 String 타입으로 변환
				.block(); // 비동기처리를 동기적으로 블록해서 결과를 반환
		return result;
		
	}
	
	
	 
//    // DB 연결 확인
//    @GetMapping("/dbtest")
//    @Cacheable("doctors")
//    public ArrayList<DoctorTb> select() {
//        return mapper.dtSelect(0);
//    }
}
