module Main(

);
    parameter native_sim = "circuitsim/intrinsics.sim:Main";
endmodule


module DataPath
    (
        input CLK,
        output [31:0] Bus, PC, IR,
        output [5:0] MicroState,
        output [31:0] reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15
    );

    reg [31:0] PC, IR, ALU_A, ALU_B, MAR;
    wire [31:0] ALU_Out, Reg_Out, Mem_Out, IR_Imm;

    wire Cmp;
    wire [3:0] OpCode;

    assign OpCode = IR[31:28];

    wire DrREG, DrMEM, DrALU, DrPC, DrOFF, LdPC;
    wire LdIR, LdMAR, LdA, LdB, LdCmp, WrREG, WrMEM;
    wire OpTest, ChkCmp;
    wire [1:0] RegSel, ALU_func;
    wire [5:0] MicroState;
    wire [3:0] RegNo;
    wire [3:0] CmpMode;
    wire [15:0] LowMAR;

    /* Interrupts */
    wire Root_IntA;
    wire PostTimer_IntA;
    wire PostKeyboard_IntA;
    wire Int;
    wire IntPullDown;
    wire [31:0] IntBus;
    wire DrDATA;
    wire LdDAR;

    wire [31:0] PC_Min;
    reg [31:0] IntAddr;

    Timer timer(CLK, Root_IntA, PostTimer_IntA, Int, IntBus);

    Keyboard key(CLK, PostTimer_IntA, PostKeyboard_IntA, Int, IntBus, IntAddr);

    ALU my_alu(ALU_A, ALU_B, ALU_func, ALU_Out);

    RegFile registers(CLK, WrREG, RegNo, Bus, Reg_Out,
                      reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15);

    Memory mem(LowMAR, Bus, CLK, WrMEM, Mem_Out);

    ControlUnit control(CLK, OpCode, Cmp,
        DrREG, DrMEM, DrALU, DrPC, DrOFF, LdPC,
        LdIR, LdMAR, LdA, LdB, LdCmp, WrREG, WrMEM,
        OpTest, ChkCmp,
        RegSel, ALU_func,
        MicroState,
        /* interrupts */
        IntPullDown,
        Root_IntA,
        DrDATA,
        LdDAR
    );

    ComparisonLogic cmp(Bus, CmpMode, Cmp);

    assign IntPullDown = Int | 1'b0;
    assign CmpMode = IR[27:24];
    assign LowMAR = MAR[15:0];
    assign IR_Imm = {IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19], IR[19:0]};

    always @(posedge CLK)
        begin
            if (LdPC)
                PC <= Bus;
            if (LdA)
                ALU_A <= Bus;
            if (LdB)
                ALU_B <= Bus;
            if (LdMAR)
                MAR <= Bus;
            if (LdIR)
                IR <= Bus;
            if (LdDAR)
                IntAddr <= Bus;

        end

    always @(*)
        begin
            case (RegSel)
                2'b00: RegNo <= IR[27:24];
                2'b01: RegNo <= IR[23:20];
                2'b10: RegNo <= IR[3:0];
                /* select $k0 */
                2'b11: RegNo <= 4'd12;
            endcase

            /* doesn't support LessThan yet lol */

            if (PC[31:4] == 0)
                PC_Min <= 32'h10;
            else
                PC_Min <= PC;


            if (DrPC)
                Bus <= PC_Min;
            if (DrALU)
                Bus <= ALU_Out;
            if (DrREG)
                Bus <= Reg_Out;
            if (DrMEM)
                Bus <= Mem_Out;
            if (DrOFF)
                Bus <= IR_Imm;
            if (DrDATA)
                Bus <= IntBus;

        end

endmodule

/* Memory is a native module loaded separately */
module Memory(
    input [15:0] Address,
    input [31:0] In,
    input Clock, Write,
    output [31:0] Out
);
    parameter native_sim = "circuitsim/intrinsics.sim:Memory";
endmodule


module Timer(
    input CLK,
    input INTA_IN,
    output INTA_OUT,
    output INT,
    output [31:0] DATA
);
    parameter native_sim = "circuitsim/int-devices.sim:Timer";
endmodule

module Keyboard(
    input CLK,
    input INTA_IN,
    output INTA_OUT,
    output INT,
    output [31:0] DATA,
    input [31:0] ADDR
);
    parameter native_sim = "circuitsim/int-devices.sim:Keyboard";
endmodule



module ControlUnit
    (
        input CLK,
        input [3:0] OpCode,
        input Cmp,

        output DrREG, DrMEM, DrALU, DrPC, DrOFF, LdPC,
        output LdIR, LdMAR, LdA, LdB, LdCmp, WrREG, WrMEM,
        output OpTest, ChkCmp,
        output [1:0] RegSel, ALU_func,
        output [5:0] State,

        /* interrupts */
        input IntTrigger,
        output IntAck,
        output DrDATA,
        output LdDAR
    );

    reg [29:0] Main_ROM [0:63];
    initial $readmem_sim("circuitsim/rom_main.dat", Main_ROM);

    reg [5:0] Seq_ROM [0:15];
    initial $readmem_sim("circuitsim/rom_seq.dat", Seq_ROM);

    reg [5:0] Cmp_ROM [0:1];
    initial $readmem_sim("circuitsim/rom_cond.dat", Cmp_ROM);

    reg [5:0] Int_ROM [0:1];
    initial $readmem_sim("circuitsim/rom_int.dat", Int_ROM);

    wire [29:0] ControlVector;

    reg [5:0] State;
    wire [5:0] NextState, NextSeqState, NextCmpState, NextIntState;

    wire [1:0] StateMux;

    /* interrupts */
    wire LdEnInt, EnInt, IntAck, DrDATA, LdDAR;
    reg IntEnable;


    assign StateMux = {ChkCmp, OpTest};

    assign DrREG = ControlVector[6];
    assign DrMEM = ControlVector[7];
    assign DrALU = ControlVector[8];
    assign DrPC = ControlVector[9];
    assign DrOFF = ControlVector[10];
    assign LdPC = ControlVector[11];

    assign LdIR = ControlVector[12];
    assign LdMAR = ControlVector[13];
    assign LdA = ControlVector[14];
    assign LdB = ControlVector[15];
    assign LdCmp = ControlVector[16];
    assign WrREG = ControlVector[17];
    assign WrMEM = ControlVector[18];

    assign RegSel = ControlVector[20:19];
    assign ALU_func = ControlVector[22:21];

    assign OpTest = ControlVector[23];
    assign ChkCmp = ControlVector[24];

    /* interrupts */
    assign LdEnInt = ControlVector[25];
    assign EnInt = ControlVector[26];
    assign IntAck = ControlVector[27];
    assign DrDATA = ControlVector[28];
    assign LdDAR = ControlVector[29];

    always @(posedge CLK)
        begin
            State <= NextState;

            if (LdEnInt)
                IntEnable <= EnInt;

        end

    always @(*)
        begin
            case (StateMux)
                2'b00: NextState <= ControlVector[5:0];
                2'b01: NextState <= NextSeqState;
                2'b10: NextState <= NextCmpState;
                2'b11: NextState <= NextIntState;
            endcase
        end

    assign ControlVector = Main_ROM[State];
    assign NextSeqState = Seq_ROM[OpCode];
    assign NextCmpState = Cmp_ROM[Cmp];
    assign NextIntState = Int_ROM[IntEnable & IntTrigger];

endmodule

module RegFile
    (
        input CLK,
        input Write,
        input [3:0] Index,
        input [31:0] In,
        output [31:0] Out,
        output [31:0] reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15
    );

    reg [31:0] reg1, reg2, reg3;
    reg [31:0] reg4, reg5, reg6, reg7;
    reg [31:0] reg8, reg9, reg10, reg11;
    reg [31:0] reg12, reg13, reg14, reg15;

    assign reg0 = 32'b0;

    always @(*)
        begin
            case (Index)
                4'b0001: Out <= reg1;
                4'b0010: Out <= reg2;
                4'b0011: Out <= reg3;
                4'b0100: Out <= reg4;
                4'b0101: Out <= reg5;
                4'b0110: Out <= reg6;
                4'b0111: Out <= reg7;
                4'b1000: Out <= reg8;
                4'b1001: Out <= reg9;
                4'b1010: Out <= reg10;
                4'b1011: Out <= reg11;
                4'b1100: Out <= reg12;
                4'b1101: Out <= reg13;
                4'b1110: Out <= reg14;
                4'b1111: Out <= reg15;
                default: Out <= 0; // zero register
            endcase
        end

    always @(posedge CLK)
        begin
            if (Write)
                begin
                    case (Index)
                        4'b0001: reg1 <= In;
                        4'b0010: reg2 <= In;
                        4'b0011: reg3 <= In;
                        4'b0100: reg4 <= In;
                        4'b0101: reg5 <= In;
                        4'b0110: reg6 <= In;
                        4'b0111: reg7 <= In;
                        4'b1000: reg8 <= In;
                        4'b1001: reg9 <= In;
                        4'b1010: reg10 <= In;
                        4'b1011: reg11 <= In;
                        4'b1100: reg12 <= In;
                        4'b1101: reg13 <= In;
                        4'b1110: reg14 <= In;
                        4'b1111: reg15 <= In;
                    endcase
                end
        end

endmodule


module ALU
    (
        input [31:0] A,
        input [31:0] B,
        input [1:0] Op,
        output [31:0] Out
    );

    always @(*)
        begin
            case (Op)
                2'b00: Out <= A+B;
                /* Out <= A - B */
                2'b01: Out <= A+ (~B) + 1;
                2'b10: Out <= ~(A & B);
                2'b11: Out <= A + 1;
            endcase
        end
endmodule


module ComparisonLogic
    (
        input [31:0] Data,
        input [3:0] Mode,
        output Out
    );

    always @(*)
        begin
            case (Mode)
                4'b0: Out <= Data == 0;
                /* Data < 0 */
                4'b1: Out <= Data[31];
                default: Out <= 0;
            endcase
        end

endmodule






