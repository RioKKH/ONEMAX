CC   = gcc
CPP  = g++
NVCC = nvcc

# サポートするアーキテクチャのリスト
SUPPORTED_ARCHS = 50 52 53 60 61 62 70 72 75 80 86

# ユーザーがARCHを指定しない場合のデフォルトアーキテクチャ
ifndef ARCH
	ARCH = 75
endif

# アーキテクチャがサポートリストに存在するか確認する
# ifeq (a, b) ... endif
#   aとbが等しい場合に、次のブロックのコードが実行される
# findstring
#   Makefileの組み込み関数で、第1引数の文字列が第2引数の文字列の中に存在するかどうかをチェックする。
# error
#   Makefileの組み込み関数で、エラーメッセージを表示してMakeの実行を停止する。
# findstringの結果が空文字列かどうかをチェックする。
ifeq (,$(findstring $(ARCH), $(SUPPORTED_ARCHS)))
	$(error Unknown architecture $(ARCH))
endif

NVCC_OPTIONS = --generate-code arch=compute_$(ARCH),code=sm_$(ARCH)

CPPFLAGS=-O3
#CPPFLAGS=-O3 -Xptxas=-v
#CPPINCLUDE=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc
CPPINCLUDE=-I./Common
#CPPINCLUDE=-I./Common  -I/usr/include/c++/9/
CFLAGS=
LDFLAGS =
debug =
#debug += -D_SHOWPOPULATION
debug += -D_SHOWRESULT
#debug= -g -D_DEBUG

APPS = gpuonemax
OBJS = CUDAKernels.o Evolution.o Parameters.o Population.o main.o

all: $(APPS)
build: $(APPS)

%.o: %.cu
	$(NVCC) $(NVCC_OPTIONS) $(debug) $(CPPINCLUDE) $(CPPFLAGS) -o $@ -c $<

$(APPS): $(OBJS)
	$(NVCC) $(LCFLAGS) $(NVCC_OPTIONS) $(debug) $(CPPINCLUDE) $(CPPFLAGS) $^ -o $@

clean:
	rm -f ${APPS}
	rm -f *.o

.PHONY: all build clean


