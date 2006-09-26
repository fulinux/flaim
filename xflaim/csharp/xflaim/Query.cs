//------------------------------------------------------------------------------
// Desc:	Db Check Status
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id$
//------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

namespace xflaim
{

	/// <summary>
	/// Flags for comparing strings.
	/// IMPORTANT NOTE: This needs to be kept in sync with the definitions in ftk.h
	/// </summary>
	[Flags]
	public enum CompareFlags : uint
	{
		/// <summary>Do case sensitive comparison.</summary>
		FLM_COMP_CASE_INSENSITIVE			= 0x0001,
		/// <summary>Compare multiple whitespace characters as a single space.</summary>
		FLM_COMP_COMPRESS_WHITESPACE		= 0x0002,
		/// <summary>Ignore all whitespace during comparison.</summary>
		FLM_COMP_NO_WHITESPACE				= 0x0004,
		/// <summary>Ignore all underscore characters during comparison.</summary>
		FLM_COMP_NO_UNDERSCORES				= 0x0008,
		/// <summary>Ignore all dash characters during comparison.</summary>
		FLM_COMP_NO_DASHES					= 0x0010,
		/// <summary>Treat newlines and tabs as spaces during comparison.</summary>
		FLM_COMP_WHITESPACE_AS_SPACE		= 0x0020,
		/// <summary>Ignore leading space characters during comparison.</summary>
		FLM_COMP_IGNORE_LEADING_SPACE		= 0x0040,
		/// <summary>Ignore trailing space characters during comparison.</summary>
		FLM_COMP_IGNORE_TRAILING_SPACE	= 0x0080,
		/// <summary>Compare wild cards</summary>
		FLM_COMP_WILD							= 0x0100
	}
}
