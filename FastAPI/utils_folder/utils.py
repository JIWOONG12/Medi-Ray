class AverageMeter(object):
    # 여러 값의 평균을 계산
    def __init__(self, name, fmt=':f'):
        self.name = name # AverageMeter의 이름 지정
        self.fmt = fmt # 출력형식 지정 format
        self.reset() # 초기화시에 reset() 호출해서 초기화

    def reset(self):
        # 초기화 하는 메서드
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1): # n : 값이 몇번 나타났는지, 기본값 1
        # 업데이트 하는 메서드
        self.val = val # val : 업데이트할 새로운 값
        self.sum += val * n # val*n을 sum에 더함
        self. count += n # count에 나타난 횟수 저장
        self.avg = self.sum / self.count # 평균
    
    def __str__(self):
        # 객체를 문자열로 표현
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # '{name}=name {val'>val값 삽입, +self.fmt > 포맷 정의 + '}({avg > avg값 삽입' + self.fmt > 포맷 정의 +'})'
        # name val (avg)
        return fmtstr.format(**self.__dict__) # 객체의 속성을 format 문자열에 대입


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches) # _get_batch_fmstr의 return 값을 batch_fmtstr에 저장
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)] # prefix + _get_batch_fmtstrt 실행된 결과 > entries의 0번 요소에 추가
        entries += [str(meter) for meter in self.meters] # self.meters에 있는 meter요소들을 문자열로 변환하여 entries에 저장
        print('\t'.join(entries)) # /t으로 구분하여 entries 출력

    def _get_batch_fmtstr(self, num_batches):
        # num_batches를 문자열로 변환후 길이를 계산 > num_digits : num_batch의 자릿수 > 공간확보
        num_digits = len(str(num_batches // 1)) # num_batches/1 > 정수로 만드는 것
        fmt = '{:' + str(num_digits) + 'd}' # fmt에 확보할 공간 지정, d : 정수형(decimal)
        # fmt : 자릿수, fmt.format(num_batches) : 포맷된 결과 (fmt의 형식에 맞게 num_batches를 변환)
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

import torch

# 모델을 저장
def save_model(name, epoch, model, optimizer,scheduler):
    torch.save({
        'epoch' : epoch + 1, # 현재 에포크를 저장
        'state_dict': model.state_dict(), # 모델의 파라피터 등을 저장
        'optimizer' : optimizer.state_dict(), # 모델의 옵티마이저 저장
        # 스케줄러 : 학습률을 동적으로 조정하는데 사용
        'scheduler' : scheduler.state_dict() # 모델의 스케줄러 저장
        }, name)
