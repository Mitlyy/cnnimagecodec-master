#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mcoder.h"

/*!
 ************************************************************************
 * \brief
 *    Initializes a given context with some pre-defined probability state
 ************************************************************************
 */
void biari_init_context 
(
	BiContextTypePtr ctx, 
	char* name
)
{
  ctx->freq_all = 0;
  for (int i=0;i<ALPHABET_SIZE;i++)
  {
	ctx->freq[i] = 1;
	ctx->freq_all+=ctx->freq[i];
  }

  ctx->cum_freq[0]=0;
  for (int i=1;i<=ALPHABET_SIZE;i++)
  {
	  ctx->cum_freq[i] = ctx->cum_freq[i-1]+ctx->freq[i-1];
  }
}

/*!
 ************************************************************************
 * \brief
 *    Allocates memory for the EncodingEnvironment struct
 ************************************************************************
 */
EncodingEnvironmentPtr arienco_create_encoding_environment()
{
  EncodingEnvironmentPtr eep;

  if ( (eep = (EncodingEnvironmentPtr) calloc(1,sizeof(EncodingEnvironment))) == NULL)
    printf("arienco_create_encoding_environment: eep");

  return eep;
}

/*!
 ************************************************************************
 * \brief
 *    Frees memory of the EncodingEnvironment struct
 ************************************************************************
 */
void arienco_delete_encoding_environment(EncodingEnvironmentPtr eep)
{
  if (eep != NULL)
  {
    free(eep);
  }
}

/*!
 ************************************************************************
 * \brief
 *    Initializes the EncodingEnvironment for the arithmetic coder
 ************************************************************************
 */
void arienco_start_encoding(EncodingEnvironmentPtr eep,
                            unsigned char *code_buffer,
                            int *code_len )
{
  eep->Elow = 0;
  eep->Erange = HALF-2;
  eep->Ebits_to_follow = 0;
  eep->Ecodestrm = code_buffer;
  eep->Ecodestrm_len = (unsigned int*)code_len;
}

/*!
 ************************************************************************
 * \brief
 *    Terminates the arithmetic codeword, writes stop bit and stuffing bytes (if any)
 ************************************************************************
 */

void arienco_done_encoding(EncodingEnvironmentPtr eep)
{
  if((eep->Elow >> (BITS_IN_REGISTER-1)) & 1)
  {
    put_one_bit_1_plus_outstanding;
  }
  else
  {
    put_one_bit_0_plus_outstanding;
  }
  
  Put1Bit(eep->Ecodestrm, *eep->Ecodestrm_len, (eep->Elow >> (BITS_IN_REGISTER-2))&1);
  Put1Bit(eep->Ecodestrm, *eep->Ecodestrm_len, 1);
    *eep->Ecodestrm_len = (*eep->Ecodestrm_len+7) & ~7;
}



void biari_encode_symbol(EncodingEnvironmentPtr eep, signed int symbol, BiContextTypePtr bi_ct)
{
  register unsigned int range = eep->Erange;
  register unsigned int low = eep->Elow;

  low = low + (range * bi_ct->cum_freq [symbol]) / bi_ct->freq_all;
  range = (range*bi_ct->freq [symbol])/bi_ct->freq_all;
  range=max(1,range);

  bi_ct->freq[symbol]++;
  bi_ct->freq_all++;
  if (bi_ct->freq_all>16384)
  {
	bi_ct->freq_all=0;
	for (int i=0;i<ALPHABET_SIZE;i++)
	{
		bi_ct->freq[i]=max(1,bi_ct->freq[i]>>1);
		bi_ct->freq_all+=bi_ct->freq[i];
	}
  }
  bi_ct->cum_freq[0]=0;
  for (int i=1;i<=ALPHABET_SIZE;i++)
  {
	bi_ct->cum_freq[i] = bi_ct->cum_freq[i-1]+bi_ct->freq[i-1];
  }

  // renormalization 
  while (range < QUARTER)
  {
    if (low >= HALF)
    {
      put_one_bit_1_plus_outstanding;
      low -= HALF;
    }
    else 
      if (low < QUARTER)
      {
        put_one_bit_0_plus_outstanding;
      }
      else
      {
        eep->Ebits_to_follow++;
        low -= QUARTER;
      }
    low <<= 1;
    range <<= 1;
  }
  
  eep->Erange = range;
  eep->Elow = low;
  
  bi_ct->freq_stat[symbol]++;
}

