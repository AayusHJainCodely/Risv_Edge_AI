#include "ch32v00x.h"
#include "debug.h"
#include "dr_inference.c"
#include "dr_model.h"

#define UART_BAUDRATE 115200
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_CHANNELS 1
#define INPUT_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)

int8_t image_buffer[INPUT_SIZE];
volatile int uart_rx_count = 0;
volatile int inference_ready = 0;

const char *class_names[NUM_CLASSES] = {
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
};

void NMI_Handler(void) __attribute__((interrupt("WCH-Interrupt-fast")));
void HardFault_Handler(void) __attribute__((interrupt("WCH-Interrupt-fast")));
void USART1_IRQHandler(void) __attribute__((interrupt("WCH-Interrupt-fast")));
void Delay_Init(void);
void Delay_Ms(uint32_t n);

void USART_Initialize(uint32_t baudrate) {
    USART_Printf_Init(baudrate);
    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
    NVIC_EnableIRQ(USART1_IRQn);
}

void USART1_IRQHandler(void) {
    if (USART_GetITStatus(USART1, USART_IT_RXNE) != RESET) {
        USART_ClearITPendingBit(USART1, USART_IT_RXNE);

        if (inference_ready) {
            (void)USART_ReceiveData(USART1);
            return;
        }

        image_buffer[uart_rx_count++] = (int8_t)USART_ReceiveData(USART1);

        if (uart_rx_count >= INPUT_SIZE) {
            uart_rx_count = 0;
            inference_ready = 1;
        }
    }
}

void DiabeticRetinopathyInference(int sample_num) {
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t  layer_in[MAX_N_ACTIVATIONS];
    int32_t prediction_index;
    uint32_t start_ticks, end_ticks;

    printf("Running inference on sample #%d...\n", sample_num);

    start_ticks = SysTick->CNT;

    processfclayer((int8_t*)image_buffer, L1_weights, L1_biases, L1_IN_NODES, L1_OUT_NODES, layer_out);
    ReLUNorm(layer_out, layer_in, L1_OUT_NODES);

    processfclayer(layer_in, L2_weights, L2_biases, L2_IN_NODES, L2_OUT_NODES, layer_out);
    ReLUNorm(layer_out, layer_in, L2_OUT_NODES);

    processfclayer(layer_in, L3_weights, L3_biases, L3_IN_NODES, L3_OUT_NODES, layer_out);
    prediction_index = findMaxIndex(layer_out, L3_OUT_NODES);

    end_ticks = SysTick->CNT;

    if (prediction_index < NUM_CLASSES) {
        printf("----------------------------------------\n");
        printf("Prediction: %ld (%s)\n", prediction_index, class_names[prediction_index]);
        printf("Timing: %lu clock cycles\n", end_ticks - start_ticks);
        printf("----------------------------------------\n\n");
    } else {
        printf("Error: Prediction index out of bounds.\n");
    }
}

int main(void) {
    SystemInit();
    Delay_Init();
    USART_Initialize(UART_BAUDRATE);

    printf("\n--- VSD Squadron Diabetic Retinopathy Classifier ---\n");
    printf("Model Input Size: %dx%d Grayscale (%d bytes)\n", IMAGE_WIDTH, IMAGE_HEIGHT, INPUT_SIZE);
    printf("Ready to receive image data via UART...\n");

    int frame_counter = 0;

    while (1) {
        if (inference_ready) {
            DiabeticRetinopathyInference(++frame_counter);
            
            inference_ready = 0;
            
            printf("Ready to receive image data via UART...\n");
        }
        
        Delay_Ms(100);
    }
}

void NMI_Handler(void) {}

void HardFault_Handler(void) {
    printf("HardFault Occurred!\n");
    while (1) {}
}
