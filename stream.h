#ifndef _STREAM_H
#define _STREAM_H

// ****************************************************
// *     Bits Order (w/o swap):                       *
// * the 1st bit in a bitstream is 0th bit (not 7th)  *
// ****************************************************

#define mask(y,nb) ((unsigned int)((int)(y) & ((1L<<(nb)) - 1)))

#ifdef TMS
#define UINT32P(x)  _mem4(x)
#else
#define UINT32P(x)  (*((unsigned int*)(x)))
#endif

#define UINT32(x) ((unsigned int)(x))
#define UCHAR(x)  ((unsigned char)(x))
#define UCHARP(x) ((unsigned char*)(x))

//-----------------------------------------------
#define PutBitsAC(stream, bitptr, x, nbits)   	\
{ 												\
   UINT32P(&(UCHARP(stream)[(bitptr)>>3]))  |=	\
           UINT32(x) << ((bitptr)&0x7);         \
   bitptr += nbits;                             \
}												\
//-----------------------------------------------
//-------------------------------------------------------------------
#define Put1BitAC(stream, bitptr, x)  								\
{       				 	  										\
  if(x) UCHARP(stream)[(bitptr)>>3] |= UCHAR(1L<<((bitptr)&0x7));	\
  (bitptr)++;                                                	  	\
}																	\
//-------------------------------------------------------------------

//---------------------------------------------------
#define PutBits(stream, bitptr, x, nbits)    		\
{                                         			\
   UINT32P(&(UCHARP(stream)[((bitptr)+7)>>3])) = 0;	\
   UINT32P(&(UCHARP(stream)[(bitptr)>>3]))    |= 	\
           UINT32(mask(x,nbits) << ((bitptr)&0x7));	\
   bitptr += nbits;									\
}													\
//---------------------------------------------------

//---------------------------------------------------
#define PutBitsOnes(stream, bitptr, nbits)       	\
{                                                   \
   UINT32P(&(UCHARP(stream)[((bitptr)+7)>>3])) = 0;	\
   UINT32P(&(UCHARP(stream)[(bitptr)>>3]))    |=    \
    UINT32(mask(0xFFFFFFFF,nbits) << ((bitptr)&0x7)); \
   bitptr += nbits;                                 \
}													\
//---------------------------------------------------
//---------------------------------------------------
#define PutBitsZeros(stream, bitptr, nbits)     	\
{                                                   \
   UINT32P(&(UCHARP(stream)[((bitptr)+7)>>3])) = 0;	\
   bitptr += nbits;                                 \
}													\
//---------------------------------------------------

//------------------------------------------------------------------
#define ShiftBitPtr(bitptr, nbits) bitptr += nbits;
//------------------------------------------------------------------

// *********************************************
// *    Put a bit into the bitstream.          *
// * x can be any value;                       *
// * only if x~=0, the current bit is set to 1 *
// *********************************************
//---------------------------------------------------------------
#define Put1Bit(stream, bitptr, x)                  			\
{                                                       		\
   UINT32P(&(UCHARP(stream)[((bitptr)+7)>>3]) ) = 0;     		\
   if((x)!=0) UINT32P(&(UCHARP(stream)[(bitptr)>>3])) |=       	\
                                UINT32(1L << ((bitptr)&0x7));	\
   (bitptr)++;                                            		\
}																\
//---------------------------------------------------------------

//-------------------------------------------------------
#define Put1BitOne(stream, bitptr)                   	\
{                                                       \
   UINT32P(&(UCHARP(stream)[((bitptr)+7)>>3]) ) = 0;	\
   UINT32P(&(UCHARP(stream)[(bitptr)>>3]) )    |=  		\
               	       UINT32(1L << ((bitptr)&0x7));	\
   (bitptr)++;                                          \
}														\
//-------------------------------------------------------
//-------------------------------------------------------
#define Put1BitZero(stream, bitptr)                   	\
{                                                       \
   UINT32P(&(UCHARP(stream)[((bitptr)+7)>>3]) ) = 0; 	\
   (bitptr)++;                                          \
}														\
//-------------------------------------------------------

//-------------------------------------------------------
#define Get1BitCABAC(stream, bitptr, x)                                 \
{                                                                  \
   x   = (stream)[(bitptr)>>3];   \
   x >>= (bitptr) & 0x7 ;                                          \
   x  &= 1;                                                        \
   (bitptr)++;                                                     \
}

//-------------------------------------------------------
#define GetBits(stream, bitptr, x, nbits)                      \
{                                                       \
   x   = UINT32P(&(UCHARP(stream)[(bitptr)>>3]));	\
   x >>= (bitptr) & 0x7 ;                               \
   bitptr += nbits;                                          \
}														\
//-------------------------------------------------------


//-------------------------------------------------------
#define Get1Bit(stream, bitptr, x)                      \
{                                                       \
   x   = UINT32P(&(UCHARP(stream)[(bitptr)>>3]));	\
   x >>= (bitptr) & 0x7 ;                               \
   x  &= 1;                                             \
   (bitptr)++;                                          \
}														\
//-------------------------------------------------------
#define Get1BitCABAC(stream, bitptr, x)                                 \
{                                                                  \
   x   = (stream)[(bitptr)>>3];   \
   x >>= (bitptr) & 0x7 ;                                          \
   x  &= 1;                                                        \
   (bitptr)++;                                                     \
}

#endif // _STREAM_H
