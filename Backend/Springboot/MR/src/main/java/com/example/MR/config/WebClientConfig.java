package com.example.MR.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {
	// web-client Bean 정의
	@Bean
	WebClient webClient() {
		// base url 지정
		// 이미지를 보냄(메모리에 보편화할 수 있는 크기 지정)
		// WebClient builder 메서드 객체가 가지고 있는 exchangeStrategies 메서드에서 코덱을 추가하여 메모리에 버퍼링 할 수 있는 크기를 무제한으로 설정
		return WebClient.builder()
				.exchangeStrategies(ExchangeStrategies.builder().codecs(configurer -> configurer.defaultCodecs()
						.maxInMemorySize(-1))
						.build())
				.baseUrl("http://localhost:8000") // 동작하고 있는 서버의 주소값
				.build();
	}
}
